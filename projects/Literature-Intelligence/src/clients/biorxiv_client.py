import requests
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import time


class BioRxivClient:
    """Client for fetching preprints from bioRxiv API."""

    BASE_URL = "https://api.biorxiv.org/details/biorxiv"

    def __init__(self):
        """Initialize bioRxiv client."""
        self.session = requests.Session()
        self.rate_limit_delay = 1.0

    def search(
        self,
        query: str,
        max_results: int = 100,
        days_back: Optional[int] = None,
        min_date: Optional[str] = None,
        max_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search bioRxiv for preprints matching query.

        Args:
            query: Search query (gene name, drug name, disease, etc.)
            max_results: Maximum number of results to return
            days_back: Number of days to look back (alternative to min_date/max_date)
            min_date: Minimum publication date (YYYY-MM-DD format)
            max_date: Maximum publication date (YYYY-MM-DD format)

        Returns:
            List of paper dictionaries with metadata
        """
        if days_back:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            min_date = start_date.strftime("%Y-%m-%d")
            max_date = end_date.strftime("%Y-%m-%d")
        elif not min_date or not max_date:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            min_date = start_date.strftime("%Y-%m-%d")
            max_date = end_date.strftime("%Y-%m-%d")

        papers = self._fetch_papers(min_date, max_date, query, max_results)
        return papers

    def _fetch_papers(
        self,
        start_date: str,
        end_date: str,
        query: str,
        max_results: int
    ) -> List[Dict[str, Any]]:
        """Fetch papers from bioRxiv API within date range."""
        papers = []

        try:
            url = f"{self.BASE_URL}/{start_date}/{end_date}/0/json"

            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            time.sleep(self.rate_limit_delay)

            data = response.json()

            if data.get('messages') and data['messages'][0].get('status') == 'ok':
                collection = data.get('collection', [])

                query_lower = query.lower()
                for item in collection:
                    if self._matches_query(item, query_lower):
                        paper = self._parse_paper(item, query)
                        if paper:
                            papers.append(paper)

                        if len(papers) >= max_results:
                            break

        except requests.exceptions.RequestException as e:
            print(f"Error fetching from bioRxiv: {e}")
        except Exception as e:
            print(f"Error processing bioRxiv response: {e}")

        return papers

    def _matches_query(self, item: Dict, query_lower: str) -> bool:
        """Check if paper matches the search query."""
        title = item.get('title', '').lower()
        abstract = item.get('abstract', '').lower()
        authors = item.get('authors', '').lower()

        return (
            query_lower in title or
            query_lower in abstract or
            query_lower in authors
        )

    def _parse_paper(self, item: Dict, query_tag: str) -> Optional[Dict[str, Any]]:
        """Parse bioRxiv paper into standardized format."""
        try:
            doi = item.get('doi', '')

            title = item.get('title', '')

            abstract = item.get('abstract', '')

            authors_str = item.get('authors', '')
            authors = [author.strip() for author in authors_str.split(';') if author.strip()]

            pub_date = item.get('date', datetime.now().strftime("%Y-%m-%d"))

            category = item.get('category', '')
            keywords = [category] if category else []

            return {
                'paper_id': doi,
                'title': title,
                'abstract': abstract,
                'authors': authors,
                'publication_date': pub_date,
                'source': 'biorxiv',
                'keywords': keywords,
                'query_tag': query_tag
            }

        except Exception as e:
            print(f"Error parsing bioRxiv paper: {e}")
            return None
