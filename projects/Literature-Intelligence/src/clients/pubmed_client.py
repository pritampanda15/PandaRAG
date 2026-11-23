from Bio import Entrez
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import time
import ssl
import urllib.request


def disable_ssl_verification():
    """Disable SSL certificate verification for environments with self-signed certificates."""
    try:
        import ssl
        _create_unverified_https_context = ssl._create_unverified_context
        ssl._create_default_https_context = _create_unverified_https_context
        print("SSL verification disabled for self-signed certificates")
    except Exception as e:
        print(f"Warning: Could not disable SSL verification: {e}")


class PubMedClient:
    """Client for fetching papers from PubMed via Entrez E-utilities API."""

    def __init__(self, email: str, api_key: Optional[str] = None, verify_ssl: bool = True):
        """
        Initialize PubMed client.

        Args:
            email: Email address for NCBI (required by Entrez)
            api_key: Optional API key for higher rate limits
            verify_ssl: Whether to verify SSL certificates (set to False for self-signed certs)
        """
        Entrez.email = email
        if api_key:
            Entrez.api_key = api_key

        if not verify_ssl:
            disable_ssl_verification()

        self.rate_limit_delay = 0.34 if api_key else 0.34

    def search(
        self,
        query: str,
        max_results: int = 100,
        days_back: Optional[int] = None,
        min_date: Optional[str] = None,
        max_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search PubMed and fetch paper metadata.

        Args:
            query: Search query (gene name, drug name, disease, etc.)
            max_results: Maximum number of results to return
            days_back: Number of days to look back (alternative to min_date/max_date)
            min_date: Minimum publication date (YYYY/MM/DD format)
            max_date: Maximum publication date (YYYY/MM/DD format)

        Returns:
            List of paper dictionaries with metadata
        """
        if days_back:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            min_date = start_date.strftime("%Y/%m/%d")
            max_date = end_date.strftime("%Y/%m/%d")

        id_list = self._search_pubmed(query, max_results, min_date, max_date)

        if not id_list:
            return []

        papers = self._fetch_details(id_list, query)
        return papers

    def _search_pubmed(
        self,
        query: str,
        max_results: int,
        min_date: Optional[str] = None,
        max_date: Optional[str] = None
    ) -> List[str]:
        """Search PubMed and return list of PMIDs."""
        try:
            search_params = {
                'db': 'pubmed',
                'term': query,
                'retmax': max_results,
                'sort': 'relevance',
                'usehistory': 'y'
            }

            if min_date and max_date:
                search_params['mindate'] = min_date
                search_params['maxdate'] = max_date
                search_params['datetype'] = 'pdat'

            handle = Entrez.esearch(**search_params)
            record = Entrez.read(handle)
            handle.close()

            time.sleep(self.rate_limit_delay)

            return record['IdList']

        except Exception as e:
            print(f"Error searching PubMed: {e}")
            return []

    def _fetch_details(self, id_list: List[str], query_tag: str) -> List[Dict[str, Any]]:
        """Fetch detailed metadata for a list of PMIDs."""
        papers = []

        try:
            ids = ','.join(id_list)
            handle = Entrez.efetch(db='pubmed', id=ids, rettype='medline', retmode='xml')
            records = Entrez.read(handle)
            handle.close()

            time.sleep(self.rate_limit_delay)

            for record in records['PubmedArticle']:
                try:
                    paper = self._parse_record(record, query_tag)
                    if paper:
                        papers.append(paper)
                except Exception as e:
                    print(f"Error parsing record: {e}")
                    continue

        except Exception as e:
            print(f"Error fetching PubMed details: {e}")

        return papers

    def _parse_record(self, record: Dict, query_tag: str) -> Optional[Dict[str, Any]]:
        """Parse a single PubMed record into standardized format."""
        try:
            article = record['MedlineCitation']['Article']
            medline = record['MedlineCitation']

            pmid = str(medline['PMID'])

            title = article.get('ArticleTitle', '')

            abstract_parts = article.get('Abstract', {}).get('AbstractText', [])
            if isinstance(abstract_parts, list):
                abstract = ' '.join([str(part) for part in abstract_parts])
            else:
                abstract = str(abstract_parts)

            authors = []
            author_list = article.get('AuthorList', [])
            for author in author_list:
                if 'LastName' in author and 'ForeName' in author:
                    authors.append(f"{author['ForeName']} {author['LastName']}")
                elif 'CollectiveName' in author:
                    authors.append(author['CollectiveName'])

            pub_date = article.get('ArticleDate', [])
            if pub_date:
                pub_date = pub_date[0]
                year = str(pub_date.get('Year', '2000'))
                month = str(pub_date.get('Month', '01')).zfill(2)
                day = str(pub_date.get('Day', '01')).zfill(2)
                date_str = f"{year}-{month}-{day}"
            else:
                journal = article.get('Journal', {}).get('JournalIssue', {}).get('PubDate', {})
                year = str(journal.get('Year', '2000'))
                month = journal.get('Month', '01')
                day = str(journal.get('Day', '01'))

                if isinstance(month, str) and month.isalpha():
                    try:
                        month = datetime.strptime(month, '%b').month
                    except:
                        month = 1

                month = str(month).zfill(2)
                day = str(day).zfill(2)
                date_str = f"{year}-{month}-{day}"

            mesh_terms = medline.get('MeshHeadingList', [])
            keywords = [str(mesh['DescriptorName']) for mesh in mesh_terms]

            return {
                'paper_id': pmid,
                'title': title,
                'abstract': abstract,
                'authors': authors,
                'publication_date': date_str,
                'source': 'pubmed',
                'keywords': keywords,
                'query_tag': query_tag
            }

        except Exception as e:
            print(f"Error parsing PubMed record: {e}")
            return None
