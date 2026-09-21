import json


def run(event, context):
    body = event.get("body")

    response = {
        "statusCode": 200,
        "body": json.dumps(body)
    }

    return response
