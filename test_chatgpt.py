import pandas as pd
import openai
import time
import os
import json
import sys
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# Configure OpenAI API key
OPENAI_API_KEY = "“

# Load QA dataset
try:
    df = pd.read_csv("UK_Immigration_31_Questions_and_Answers_Categorized.csv")
    print(f"Loaded {len(df)} QA pairs")
except FileNotFoundError:
    print("Error: File 'uk_immigration_qa_from_content.csv' not found")
    sys.exit(1)

# Initialize OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# ChatGPT interaction function
def get_chatgpt_response(prompt, max_retries=5):
    """Get ChatGPT response with timeout and retries"""
    for attempt in range(max_retries):
        try:
            print(f"Sending request to OpenAI API (Attempt {attempt+1})...")

            start_time = time.time()
            response = client.chat.completions.create(
                model="chatgpt-4o-latest",
                messages=[
                    {"role": "system",
                     "content": "You are an expert on UK immigration law. Provide accurate, concise answers based on official UK government information."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=400,
                top_p=0.9,
                timeout=20  # Timeout for the API call
            )
            response_time = time.time() - start_time

            print(f"Response time: {response_time:.2f} seconds")
            content = response.choices[0].message.content.strip()
            print(f"Response content: {content[:100]}...")
            return content

        except openai.RateLimitError as e:
            print(f"Rate limit reached: {e}. Retrying in 10s...")
            time.sleep(10)
        except openai.APIError as e:
            if 500 <= e.status_code < 600:
                print(f"Server error: {e}. Retrying in 10s...")
                time.sleep(10)
            else:
                print(f"OpenAI API error: {e}")
                return f"Error: {e}"
        except openai.Timeout as e:
            print(f"Timeout error: {e}. Retrying in 10s...")
            time.sleep(10)
        except Exception as e:
            print(f"Unexpected error: {e}. Retrying in 5s...")
            time.sleep(5)

    return "Error: Failed after multiple retries."


# Evaluation functions
def calculate_similarity(answer1, answer2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([answer1, answer2])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]


def evaluate_answer(model_response, correct_answer):
    similarity = calculate_similarity(model_response, correct_answer)

    key_metrics = {
        "contains_amount": any(char in model_response for char in ['£', '$', '€']) and any(
            char in correct_answer for char in ['£', '$', '€']),
        "contains_time": any(word in model_response.lower() for word in ['day', 'week', 'month', 'year']) and any(
            word in correct_answer.lower() for word in ['day', 'week', 'month', 'year']),
        "contains_condition": any(
            word in model_response.lower() for word in ['must', 'require', 'need', 'condition']) and any(
            word in correct_answer.lower() for word in ['must', 'require', 'need', 'condition'])
    }

    score = min(5, similarity * 5)

    key_points_covered = sum(key_metrics.values())
    score = min(5, score + key_points_covered * 0.5)

    return {
        "similarity": similarity,
        "score": round(score, 2),
        "key_metrics": key_metrics,
        "key_points_covered": key_points_covered
    }


# Run test
results = []

print(f"Starting ChatGPT test on {len(df)} questions...")

log_dir = "chatgpt_test_logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"test_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
results_csv = os.path.join(log_dir, "chatgpt_uk_immigration_test_results.csv")
results_json = os.path.join(log_dir, "chatgpt_uk_immigration_test_results.json")

with open(log_file, "w", encoding="utf-8") as log:
    log.write(f"ChatGPT API Test Log - {datetime.now()}\n")
    log.write(f"API Key: {OPENAI_API_KEY}\n")
    log.write(f"Total Questions: {len(df)}\n\n")

    for index, row in df.iterrows():
        question = row['question']
        correct_answer = row['answer']
        category = row['category']
        ##source = row['Source']

        print(f"\nTesting Question {index + 1}/{len(df)}: {question[:80]}...")
        log.write(f"\n[Question {index + 1}/{len(df)}] {question}\n")

        try:
            print("Fetching ChatGPT response...")
            log.write("Fetching ChatGPT response...\n")

            chatgpt_response = get_chatgpt_response(question)
            time.sleep(1)

            print(f"ChatGPT response: {chatgpt_response[:100]}...")
            log.write(f"ChatGPT response: {chatgpt_response}\n")
        except Exception as e:
            print(f"API call error: {str(e)}")
            log.write(f"API call error: {str(e)}\n")
            chatgpt_response = f"Error: {str(e)}"

        evaluation = evaluate_answer(chatgpt_response, correct_answer)

        result = {
            "question_id": index,
            "question": question,
            "correct_answer": correct_answer,
            "category": category,
            #"source": source,
            "chatgpt_response": chatgpt_response,
            "similarity": evaluation["similarity"],
            "score": evaluation["score"],
            "key_points_covered": evaluation["key_points_covered"]
        }

        results.append(result)
        log.write(
            f"Evaluation: Score={evaluation['score']}/5, Similarity={evaluation['similarity']:.3f}, Key Points Covered={evaluation['key_points_covered']}/3\n")

        print(f"Score: {evaluation['score']}/5 | Similarity: {evaluation['similarity']:.3f} | Key Points: {evaluation['key_points_covered']}/3")

    results_df = pd.DataFrame(results)
    results_df.to_csv(results_csv, index=False)

    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved:")
    print(f"  - CSV: {results_csv}")
    print(f"  - JSON: {results_json}")
    log.write(f"\nResults saved to:\n  - {results_csv}\n  - {results_json}")


# Generate summary
def generate_summary_report(results_df):
    report = {
        "total_questions": len(results_df),
        "test_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "model": "gpt-4-turbo"
    }

    report["average_score"] = results_df["score"].mean()
    report["min_score"] = results_df["score"].min()
    report["max_score"] = results_df["score"].max()
    report["average_similarity"] = results_df["similarity"].mean()
    report["average_key_points"] = results_df["key_points_covered"].mean()

    category_scores = []
    for category in df['category'].unique():
        cat_df = results_df[results_df['category'] == category]
        if len(cat_df) > 0:
            category_scores.append({
                "category": category,
                "count": len(cat_df),
                "average_score": cat_df["score"].mean(),
                "average_similarity": cat_df["similarity"].mean(),
                "average_key_points": cat_df["key_points_covered"].mean()
            })

    report_df = pd.DataFrame({
        "Metric": [
            "Total Questions",
            "Average Score",
            "Min Score",
            "Max Score",
            "Average Similarity",
            "Average Key Points Covered"
        ],
        "Value": [
            report["total_questions"],
            f"{report['average_score']:.2f}",
            f"{report['min_score']:.2f}",
            f"{report['max_score']:.2f}",
            f"{report['average_similarity']:.3f}",
            f"{report['average_key_points']:.2f}/3"
        ]
    })

    category_report_df = pd.DataFrame(category_scores)

    report_xlsx = os.path.join(log_dir, "chatgpt_uk_immigration_test_summary.xlsx")
    with pd.ExcelWriter(report_xlsx) as writer:
        report_df.to_excel(writer, sheet_name="Summary", index=False)
        category_report_df.to_excel(writer, sheet_name="Category Scores", index=False)

    plt.figure(figsize=(10, 6))
    sns.histplot(results_df["score"], bins=10, kde=True)
    plt.title("ChatGPT Score Distribution")
    plt.xlabel("Score (0-5)")
    plt.ylabel("Number of Questions")
    plt.axvline(report["average_score"], color='r', linestyle='--', label=f'Average: {report["average_score"]:.2f}')
    plt.legend()
    plt.tight_layout()
    score_distribution_png = os.path.join(log_dir, "chatgpt_score_distribution.png")
    plt.savefig(score_distribution_png)

    plt.figure(figsize=(12, 6))
    if category_scores:
        plot_data = []
        for cat in category_scores:
            plot_data.append({"Category": cat["category"], "Score": cat["average_score"]})

        plot_df = pd.DataFrame(plot_data)
        sns.barplot(x="Category", y="Score", data=plot_df, palette="viridis")
        plt.title("ChatGPT Performance by Question Category")
        plt.ylabel("Average Score")
        plt.xlabel("Category")
        plt.ylim(0, 5)
        plt.tight_layout()
        performance_by_category_png = os.path.join(log_dir, "chatgpt_performance_by_category.png")
        plt.savefig(performance_by_category_png)

    print("\n" + "=" * 50)
    print("ChatGPT Test Summary Report")
    print("=" * 50)
    print(f"Total Questions: {report['total_questions']}")
    print(f"Average Score: {report['average_score']:.2f}/5")
    print(f"Min Score: {report['min_score']:.2f}/5")
    print(f"Max Score: {report['max_score']:.2f}/5")
    print(f"Average Similarity: {report['average_similarity']:.3f}")
    print(f"Average Key Points Covered: {report['average_key_points']:.2f}/3")

    if category_scores:
        print("\nPerformance by Category:")
        for cat in category_scores:
            print(
                f"  - {cat['category']}: {cat['average_score']:.2f}/5 (Similarity: {cat['average_similarity']:.3f}, Key Points: {cat['average_key_points']:.2f}/3)")

    print("\nReport generated:")
    print(f"  - Excel report: {report_xlsx}")
    print(f"  - Score distribution chart: {score_distribution_png}")
    if category_scores:
        print(f"  - Category performance chart: {performance_by_category_png}")

    return report


# Generate report
generate_summary_report(pd.DataFrame(results))

print("\nChatGPT Test Completed.")
print(f"Log file: {log_file}")
