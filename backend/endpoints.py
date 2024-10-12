
import flask as Flask
import requests
from helpers import Finance, WebScrape, Groq, Neo4j

app = Flask(__name__)

@app.route("/api/stock_analysis", ['GET'])
def stock_name():
    data = requests.json()
    stock = data['ticker'] 

    financial_data = Finance.get_finance_data(stock)
    all_links_combined = WebScrape.top_links(stock)
    whole_text = WebScrape.return_useful_text(all_links_combined)
    graph = Neo4j.init_graph()
    Neo4j.process_document(whole_text, stock, graph, chunk_size=700, chunk_overlap=100)

    print("Your knowledge graph about ", stock, " Is Completed :)")


if __name__ == "__main__":
    app.run(port = 5013, debug = True, host = '0.0.0.0')