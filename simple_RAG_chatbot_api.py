from flask import Flask, request, jsonify
from sentence_transformers import CrossEncoder
from llmchain import chain
from vectorstore import db

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

app = Flask(__name__)

@app.route('/api/chat', methods=['POST'])
def chat():
    try:

        data = request.get_json()

        # Validate request body
        if not data:
            return jsonify({
                'error': 'Request body must be JSON'
            }), 400

        # Validate required field
        user_message = data.get('query')

        if not user_message:
            return jsonify({
                'error': 'Missing required field: query'
            }), 400

        # Retrieve documents
        docs = db.similarity_search(
            user_message,
            k=10
        )

        # Prepare query-doc pairs
        pairs = [
            [user_message, doc.page_content]
            for doc in docs
        ]

        # Get reranking scores
        scores = reranker.predict(pairs)

        # Combine docs + scores
        scored_docs = list(zip(docs, scores))

        # Sort by relevance
        scored_docs = sorted(
            scored_docs,
            key=lambda x: x[1],
            reverse=True
        )

        # Keep top reranked docs
        reranked_docs = [
            doc for doc, score in scored_docs[:3]
        ]

        # Join context into text
        context = "\n\n".join({
            doc.page_content
            for doc in reranked_docs
        })


        # Invoke LLM chain
        response = chain.invoke({
            'query': user_message,
            'context':context
            })

        return jsonify({
            'response': response
        })

    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
