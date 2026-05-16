from flask import Flask, request, jsonify
from langchain_core.messages import HumanMessage
from multi_agent import graph

app = Flask(__name__)


@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "error": "Request body must be JSON"
            }), 400

        user_query = data.get("query")

        if not user_query:
            return jsonify({
                "error": "Missing required field: query"
            }), 400

        response = graph.invoke({
            "messages": [
                HumanMessage(content=user_query)
            ]
        })

        final_response = {
            "query": user_query,
            "research_data": response.get("research_data", ""),
            "analysis": response.get("analysis", ""),
            "final_report": response.get("final_report", ""),
            "task_complete": response.get("task_complete", False),
            "messages": [
                msg.content for msg in response.get("messages", [])
            ]
        }

        return jsonify({
            "success": True,
            "response": final_response
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
