"""
Guardian API Fetcher
Fetches articles from Guardian API and extracts verifiable claims
"""

import requests
from typing import List, Dict
from claim_extractor import ClaimExtractor


class GuardianFetcher:
    """Fetches articles from Guardian API and extracts claims"""
    
    def __init__(self, api_key: str):
        """
        Initialize Guardian API fetcher
        
        Args:
            api_key: Guardian API key (get from https://open-platform.theguardian.com/)
        """
        self.api_key = api_key
        self.base_url = "https://content.guardianapis.com/search"
        self.claim_extractor = ClaimExtractor()
    
    def fetch_articles(self, 
                      from_date: str, 
                      to_date: str, 
                      page_size: int = 200,
                      max_pages: int = 5) -> List[Dict]:
        """
        Fetch articles from Guardian API
        
        Args:
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            page_size: Number of articles per page (max 200)
            max_pages: Maximum number of pages to fetch
            
        Returns:
            List of article dictionaries in claim_extractor format
        """
        articles = []
        
        for page in range(1, max_pages + 1):
            print(f"Fetching page {page}...")
            
            # API parameters
            params = {
                'from-date': from_date,
                'to-date': to_date,
                'order-by': 'newest',
                'show-fields': 'bodyText',  # Get full article text
                'page-size': page_size,
                'page': page,
                'api-key': self.api_key
                # 'q': (
                #     "inflation OR economy OR interest rates OR recession OR GDP OR unemployment OR "
                #     "election OR Congress OR legislation OR policy OR government OR foreign policy OR "
                #     "technology OR AI OR cybersecurity OR innovation OR research OR science OR medicine OR "
                #     "climate OR emissions OR energy OR environment OR public health OR pandemic OR "
                #     "geopolitics OR diplomacy OR US OR global"
                # )
            }
            
            try:
                response = requests.get(self.base_url, params=params)
                response.raise_for_status()
                data = response.json()
                
                # Extract articles from response
                results = data.get('response', {}).get('results', [])
                
                if not results:
                    print(f"No more articles found on page {page}")
                    break
                
                # Format articles for claim extractor
                for article in results:
                    formatted_article = {
                        'webPublicationDate': article.get('webPublicationDate', ''),
                        'fields': {
                            'bodyText': article.get('fields', {}).get('bodyText', '')
                        }
                    }
                    articles.append(formatted_article)
                
                print(f"Fetched {len(results)} articles from page {page}")
                
            except requests.exceptions.RequestException as e:
                print(f"Error fetching page {page}: {e}")
                break
        
        print(f"\nTotal articles fetched: {len(articles)}\n")
        return articles
    
    def fetch_and_extract_claims(self,
                                 from_date: str,
                                 to_date: str,
                                 page_size: int = 200,
                                 max_pages: int = 5) -> List[str]:
        """
        Fetch articles and extract all verifiable claims
        
        Args:
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            page_size: Number of articles per page
            max_pages: Maximum number of pages to fetch
            
        Returns:
            List of formatted claim strings: ["claim | date", ...]
        """
        # Fetch articles
        articles = self.fetch_articles(from_date, to_date, page_size, max_pages)
        
        # Extract claims from all articles
        all_claims = []
        print("Extracting claims from articles...")
        
        for i, article in enumerate(articles, 1):
            print(f"Processing article {i}/{len(articles)}...", end='\r')
            
            claims = self.claim_extractor.extract_from_guardian_article(article)
            
            # Convert to formatted strings
            formatted_claims = [c['formatted'] for c in claims]
            all_claims.extend(formatted_claims)
        
        print(f"\n\nTotal claims extracted: {len(all_claims)}\n")
        return all_claims


def main():
    """Example usage"""
    
    # Get API key from user
    print("="*70)
    print("GUARDIAN API FETCHER & CLAIM EXTRACTOR")
    print("="*70)
    print("\nGet a free API key at: https://open-platform.theguardian.com/")
    api_key = input("Enter your Guardian API key: ").strip()
    
    if not api_key:
        print("Error: API key required")
        return
    
    # Get date range
    print("\nEnter date range for articles (YYYY-MM-DD format)")
    from_date = input("From date (e.g., 2025-01-01): ").strip()
    to_date = input("To date (e.g., 2025-01-31): ").strip()
    
    # Initialize fetcher
    fetcher = GuardianFetcher(api_key)
    
    # Fetch articles and extract claims
    claims = fetcher.fetch_and_extract_claims(
        from_date=from_date,
        to_date=to_date,
        page_size=50,  # Start small for testing
        max_pages=2    # Just 2 pages for testing
    )
    
    # Display results
    print("="*70)
    print("EXTRACTED CLAIMS")
    print("="*70)
    print("\nFirst 10 claims:")
    for i, claim in enumerate(claims[:10], 1):
        print(f"{i}. {claim}")
    
    if len(claims) > 10:
        print(f"\n... and {len(claims) - 10} more claims")
    
    # Save to file
    output_file = "guardian_claims.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        for claim in claims:
            f.write(claim + '\n')
    
    print(f"\nAll claims saved to: {output_file}")
    print(f"\nReady to use in RAG! Load this file or use the 'claims' list directly.")


if __name__ == "__main__":
    main()
