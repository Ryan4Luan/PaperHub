import arxiv
import json
from typing import Generator


def create_arxiv_client() -> arxiv.Client:
    """
    Constructs and returns the default arXiv API client.

    Returns:
        arxiv.Client: The arXiv API client.
    """
    return arxiv.Client()


def search_arxiv(client: arxiv.Client, query: str, max_results: int = 10) -> Generator[arxiv.Result, None, None]:
    """
    Searches arXiv for the most recent articles matching the given query.

    Args:
        client (arxiv.Client): The arXiv API client.
        query (str): The search query.
        max_results (int): The maximum number of results to return.

    Returns:
        Generator[arxiv.Result, None, None]: A generator yielding search results.

    Example:
        client = arxiv.Client()
        results = search_arxiv(client, query="machine learning", max_results=5)
        for result in results:
            print(result.title)
    """
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    return client.results(search)


def save_arxiv_results_as_json(results: Generator[arxiv.Result, None, None], file_name: str) -> None:
    """
    Saves information about each result in the provided generator to a JSON file.

    Args:
        results (Generator[arxiv.Result, None, None]): The generator yielding search results.
        file_name (str): The name of the output JSON file.
    """
    data = []

    for r in results:
        result_data = {
            "title": r.title,
            "authors": [author.name for author in r.authors],
            "summary": r.summary,
            "published": r.published.isoformat(),
            "doi": r.doi,
            "pdf_link": r.pdf_url,
            "arxiv_link": r.entry_id,
            "categories": r.categories,
            "journal_ref": r.journal_ref if r.journal_ref else "N/A"
        }
        data.append(result_data)

    with open(file_name, 'w') as f:
        json.dump(data, f, indent=4)


def main():
    """
    Main function to execute the arXiv search and print the results.
    """
    # Create an arXiv client
    client = arxiv.Client()

    # Search for the most recent articles matching the query "machine learning"
    results = search_arxiv(client, query="machine learning", max_results=5)

    # Save the results to a JSON file
    save_arxiv_results_as_json(results, "./arxiv_results.json")


if __name__ == "__main__":
    main()

# # ...or exhaust it into a list. Careful: this is slow for large results sets.
# all_results = list(results)
# print([r.title for r in all_results])
#
# # For advanced query syntax documentation, see the arXiv API User Manual:
# # https://arxiv.org/help/api/user-manual#query_details
# search = arxiv.Search(query="au:Kaiming He AND ti:resnet")
# first_result = next(client.results(search))
# print(first_result)
#
# # Search for the paper with ID "1605.08386v1"
# search_by_id = arxiv.Search(id_list=["1605.08386v1"])
# # Reuse client to fetch the paper, then print its title.
# first_result = next(client.results(search))
# print(first_result.title)
