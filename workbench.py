# wsa_a15b8a7b55638968db49a3b3ccc23ec2a344ee8db67acb49f8d49b9876b28918
{
  "searchParameters": {
    "query": "What is python",
    "maxResults": 5,
    "includeContent": false,
    "country": "us",
    "language": "en"
  },
  "organic": [
    {
      "title": "What is Python? Executive Summary",
      "url": "https://www.python.org/doc/essays/blurb/",
      "description": "Python is an interpreted, object-oriented, high-level programming language with dynamic semantics. Its high-level built in data structures, combined with ...",
      "position": 1
    },
    {
      "title": "Welcome to Python.org",
      "url": "https://www.python.org/",
      "description": "Python is a programming language that lets you work quickly and integrate systems more effectively. Learn More",
      "position": 2
    },
    {
      "title": "What Is Python Used For? A Beginner's Guide - Coursera",
      "url": "https://www.coursera.org/articles/what-is-python-used-for-a-beginners-guide-to-using-python",
      "description": "What is Python? Python is a computer programming language often used to build websites and software, automate tasks, and conduct data analysis.",
      "position": 3
    },
    {
      "title": "Python (programming language) - Wikipedia",
      "url": "https://en.wikipedia.org/wiki/Python_(programming_language)",
      "description": "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation.",
      "position": 4
    },
    {
      "title": "What is Python? - Python Language Explained - AWS",
      "url": "https://aws.amazon.com/what-is/python/",
      "description": "Python is a programming language that is widely used in web applications, software development, data science, and machine learning (ML).",
      "position": 5
    }
  ]
}

import requests
import json

url = "https://api.websearchapi.ai/ai-search"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer wsa_a15b8a7b55638968db49a3b3ccc23ec2a344ee8db67acb49f8d49b9876b28918"
}

payload = {
    "query": "What is python",
    "maxResults": 5,
    "includeContent": False,
    "country": "us",
    "language": "en"
}

response = requests.post(url, headers=headers, json=payload)

# Check if the request was successful
if response.status_code == 200:
    results = response.json()
    print(json.dumps(results, indent=2))
else:
    print(f"Error: {response.status_code}")
    print(response.text)