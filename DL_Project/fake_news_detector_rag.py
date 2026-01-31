import numpy as np
import torch
import kMeansRAG
import example_evidence
import random
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score, classification_report
from tqdm import tqdm

# ============================================================================
# Import Guardian fetcher and Fact Checker LLM
# ============================================================================
from guardian_fetcher import GuardianFetcher
from fact_checker_llm import FactCheckerLLM

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
#reranker = CrossEncoder("cross-encoder/nli-deberta-v3-base", device="cuda")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2", device="cuda")
pca = PCA(n_components=.90, svd_solver="full", random_state=1)

# ============================================================================
# Initialize Fact Checker LLM
# ============================================================================
fact_checker = FactCheckerLLM()
used_chunks = []

# special features on this code: kMeans++, PCA, Early Hop/Top_N Stopping

class Chunk:
    def __init__(self, id, evidence, embedding):
        self.id = id
        self.parent_id = None
        self.evidence = evidence
        self.embedding = embedding
        self.probability = None
        self.cosine_similarity = None
        self.rerank_score = 0.0
        self.visited = False

def printTopEmbeddings(topEmbeddings, var):
    print("top {} embeddings:".format(var))
    for x in topEmbeddings:
        print(x.evidence + " ", end="")
    print()

def L2norm(vec):
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec

def limitTopN(top_list, evidence, top_n=20):
    top_list.append(evidence)
    return sorted(top_list, key=lambda x: x.cosine_similarity, reverse=True)[:top_n]

def topNclaims(input_embedding, evidence_data, top_n, cutoff):
    top_N_similarities = []
    for evidence in evidence_data:
        if evidence.visited:
            continue
        if evidence.cosine_similarity is None:
            evidence.cosine_similarity = np.dot(input_embedding, evidence.embedding)
        top_N_similarities = limitTopN(top_N_similarities, evidence, top_n)
    best_N = []
    for n in top_N_similarities:
        # if candidate is not at least 80% relevant compared to the best value in TOP_N, then terminate
        if n.cosine_similarity > top_N_similarities[0].cosine_similarity * cutoff:
            best_N.append(n)
        else:
            break
    #printTopEmbeddings(best_N, "N")
    return best_N

def topKclaims(input_text, candidates, top_k):
    context_addition = ""
    rerank_cands = []
    for cand in candidates:
        rerank_cands.append((input_text, cand.evidence))
    with torch.inference_mode():
        reranked_score = reranker.predict(rerank_cands)
    for i in range(len(candidates)):
        candidates[i].rerank_score = reranked_score[i]
    top_K_similarities = sorted(candidates, key=lambda x: x.rerank_score, reverse=True)[:top_k]
    #printTopEmbeddings(top_K_similarities, "K")
    for cand in top_K_similarities:
        context_addition += "\n" + cand.evidence
        cand.visited = True
        used_chunks.append(cand)
    return context_addition

def bestClusters(input_query_embedding, cluster_map, top_x):
    best_clusters = []
    retrieved_evidence = []
    for key, val in cluster_map.items():
        best_clusters.append((key, np.dot(input_query_embedding, val.embedding_avg)))
    best_clusters = sorted(best_clusters, key=lambda x: x[1], reverse=True)[:top_x]
    #print("Best clusters in descending order of cosine similarity:")
    for id, _ in best_clusters:
        #print("Best cluster id: {}".format(id))
        for point in cluster_map[id].points:
            #print("Evidence: {}".format(point.evidence))
            retrieved_evidence.append(point)
    #print()
    return retrieved_evidence

# top n is amount of chunks gathered quickly, top k is amount of chunks gathered in depth
# will be at least cutoff% as good as previous hop
def gatherInformation(input_text, input_embedding, evidence_data, top_n, top_k, num_hops=3, cutoff=0.8):
    context = ""
    prev_score = float("-inf")
    for hop in range(num_hops):
        candidates = topNclaims(input_embedding, evidence_data, top_n, cutoff)
        #if len(candidates) > 0 and candidates[0].cosine_similarity > prev_score * cutoff:
        context += topKclaims(input_text, candidates, top_k)
        #prev_score = candidates[0].cosine_similarity
            #print("Updated Context:\n", context, "\n")
        #else:
            # print("----------------------------------------------------")
            # print("Stopping Early. Next Hop information is not relevant.")
            # print("----------------------------------------------------")
            #break
    return context




if __name__ == "__main__":
    # top X: look only in top X clusters
    # top N: quick search of clusters, gather top N samples from X clusters
    # top K: deep search to rerank top N samples and take best K
    # num hops: amount of times program is ran. New context is added every search if context available.
    # num clusters: amount of groupings formed from clustering algorithm.
    # cutoff: worst Top N and Hop cand will be at least best_n or prev_hop * (cutoff/100)
    TOP_X = 8
    TOP_N = 1000
    TOP_K = 50
    NUM_HOPS = 3
    NUM_CLUSTERS = 8 # speed vs quality. High number, high quality and slower. Low number, lower quality and faster.
                      # with High number, less possible results --> more narrow. with Low number, more possible results --> more broad.
    CUTOFF = 0.4
    # equation for how much of graph will be searched average case:
    # y = ((TOP_X * len(global_evidence_database) / NUM_CLUSTERS) + NUM_CLUSTERS) / (len(global_evidence_database) + NUM_CLUSTERS)
    # min value of y is best num_cluster for efficiency


    # global_evidence_embedding = model.encode(global_evidence_strings, normalize_embeddings=True)
    data = np.load("embedded_evidence.npz", allow_pickle=True)
    global_evidence_strings = data["texts"].tolist()
    global_evidence_embedding = data["embeddings"]
    #global_evidence_embedding = pca.fit_transform(global_evidence_embedding)
    global_evidence_database = []
    for i in range(len(global_evidence_strings)):
        global_evidence_database.append(Chunk(i+(NUM_CLUSTERS * 100), global_evidence_strings[i], L2norm(global_evidence_embedding[i])))

    # create clusters.
    #seedList = random.sample(global_evidence_database, NUM_CLUSTERS)
    print("starting kMeansPlusPlus")
    seedList = kMeansRAG.kMeansPlusPlus(global_evidence_database, NUM_CLUSTERS)
    clusterMap = kMeansRAG.kMeans(global_evidence_database, seedList=seedList)
    print("kMeansPlusPlus done")

    # print clusters
    # for key, val in clusterMap.items():
    #     print("Center id: {}".format(key))
    #     for point in val.points:
    #         print("point id: ({}) --> {}".format(point.id, point.evidence))
    # print()

    # get user input and pass into sentence transformer.
    # input_query = input("\nEnter news to verify: ").strip()
    # if not input_query:
    #     input_query = "Does Donald Trump own a pet monkey?"
    #     print(f"Using example query: {input_query}")

    testing_set = np.load("claims_with_label.npz", allow_pickle=True)
    inputs_set = testing_set["inputs"]
    labels_set = testing_set["labels"]

    inputs_set = inputs_set[:300]
    labels_set = labels_set[:300]
    llm_rag_results = []
    BATCH_SIZE = 10
    #for input_query in inputs:
    for start in tqdm(range(0, len(inputs_set), BATCH_SIZE)):
        #input_query = inputs_set[inp_q]
        batch_inputs = inputs_set[start:start+BATCH_SIZE]
        batch_contexts = []
        for input_query in batch_inputs:
            # reset chunk values.
            for chunk in used_chunks:
                chunk.cosine_similarity = None
                chunk.rerank_score = 0.0
                chunk.visited = False
            used_chunks.clear()

            # take in new input query
            with torch.inference_mode():
                input_query_embedding = model.encode(input_query, normalize_embeddings=True)
            #input_query_embedding = pca.transform(input_query_embedding.reshape(1, -1))[0]
            input_query_embedding = L2norm(input_query_embedding)

            # RAG PIPELINE
            retrieved_evidence = bestClusters(input_query_embedding, clusterMap, TOP_X)
            llm_context = gatherInformation(input_query, input_query_embedding, retrieved_evidence, TOP_N, TOP_K, NUM_HOPS, CUTOFF)
            llm_context = llm_context[:1000]
            batch_contexts.append(llm_context)
            print("-----------------------------------------------------")
            print("claim: ", input_query)
            print("llm context: ", llm_context)
            print("-----------------------------------------------------")
            y = (TOP_X * (len(global_evidence_database) / NUM_CLUSTERS) + NUM_CLUSTERS) / (len(global_evidence_database) + NUM_CLUSTERS) * 100
            #print("Input Query:\n", input_query)
            #print("LLM Context:\n", llm_context)
            #print("\nSearch only needed to examine {}% of the graph.".format(round(y,2)))

            # ============================================================================
            # Verify news with Fact Checker LLM
            # ============================================================================
            # print("\n" + "="*70)
            # print("VERIFYING NEWS WITH FACT CHECKER LLM")
            # print("="*70 + "\n")


        with torch.inference_mode():
            result = fact_checker.verify_news_batch(batch_inputs, batch_contexts)

        # print("normal result", result)
        # print("result40:", result[:40])

        for res in result:
            print(res)
            if "SUPPORT" in res:
                #"VERDICT: REFUTED" in result[:40]:
                llm_rag_results.append("SUPPORTS")
                #print(res)
            elif "REFUTE" in res:
                llm_rag_results.append("REFUTES")
            else:
                # print("odd one neither?")
                # print(res)
                llm_rag_results.append("NOT ENOUGH INFO")
        # else:
        #     llm_rag_results.append("NOT ENOUGH INFO")

    # print("LABELS RESULTS: ", labels_set)
    # print("LLM RESULTS: ", llm_rag_results)
    report = classification_report(labels_set, llm_rag_results, zero_division=0)
    print(report)

    np_llm_rag_results = np.array(llm_rag_results, dtype=object)


#               precision    recall  f1-score   support
#
#      REFUTES       0.54      0.98      0.70       162
#     SUPPORTS       0.62      0.04      0.07       138
#
#     accuracy                           0.55       300
#    macro avg       0.58      0.51      0.38       300
# weighted avg       0.58      0.55      0.41       300


#               precision    recall  f1-score   support
#
#      REFUTES       0.67      1.00      0.80         6
#     SUPPORTS       1.00      0.25      0.40         4
#
#     accuracy                           0.70        10
#    macro avg       0.83      0.62      0.60        10
# weighted avg       0.80      0.70      0.64        10


# eval todo:
# show what the dataset looks like in an image or example or write about ti
# evaluation stuff images about results
# challenges section
# conclusion section