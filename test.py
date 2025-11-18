from src import search_google_scholar_articles, delete_temp_results_dir

result = search_google_scholar_articles("graph neural networks", max_results=5)
print(result.articles)
print(result.json_path)

delete_temp_results_dir(result.temp_dir)