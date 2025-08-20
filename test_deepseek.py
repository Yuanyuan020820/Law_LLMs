import pandas as pd
import requests
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

# Configure DeepSeek API key

# Load QA dataset
df = pd.read_csv("UK_Immigration_31_Questions_and_Answers_Categorized.csv")
print(f"Loaded {len(df)} QA pairs")


# DeepSeek model response function
def get_deepseek_response(prompt):
    """Get response from DeepSeek API"""
    try:
        url = "https://api.deepseek.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system",
                 "content": "You are an expert on UK immigration law. Provide accurate, concise answers based on official UK government information."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 400,
            "top_p": 0.9
        }
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"DeepSeek error: {str(e)}")
        return f"Error: {str(e)}"


# Evaluation functions
def calculate_similarity(answer1, answer2):
    """Calculate cosine similarity using TF-IDF"""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([answer1, answer2])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]


def evaluate_answer(model_response, correct_answer):
    """Evaluate model response against reference answer"""
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


# Run evaluation
results = []
print(f"Starting DeepSeek model evaluation on {len(df)} questions...")

for index, row in df.iterrows():
    question = row['question']
    correct_answer = row['answer']
    category = row['category']
    #source = row['source']

    print(f"\nEvaluating question {index + 1}/{len(df)}: {question[:80]}...")

    try:
        print("  Getting response from DeepSeek...")
        deepseek_response = get_deepseek_response(question)
        time.sleep(1)
    except Exception as e:
        print(f"  API error: {str(e)}")
        deepseek_response = "Error: API call failed"

    deepseek_eval = evaluate_answer(deepseek_response, correct_answer)

    result = {
        "question_id": index,
        "question": question,
        "correct_answer": correct_answer,
        "category": category,
    #    "source": source,
        "deepseek_response": deepseek_response,
        "deepseek_similarity": deepseek_eval["similarity"],
        "deepseek_score": deepseek_eval["score"],
        "key_points_covered": deepseek_eval["key_points_covered"]
    }

    results.append(result)
    print(
        f"  Score: {deepseek_eval['score']}/5 | Similarity: {deepseek_eval['similarity']:.2f} | Key Points: {deepseek_eval['key_points_covered']}/3")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv("deepseek_uk_immigration_test_results.csv", index=False)
print("\nResults saved to deepseek_uk_immigration_test_results.csv")

os.makedirs("deepseek_test_logs", exist_ok=True)
with open("deepseek_test_logs/deepseek_uk_immigration_test_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("Results saved to deepseek_uk_immigration_test_results.json")


# Summary report generation
def generate_summary_report(results_df):
    report = {
        "total_questions": len(results_df),
        "test_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "model": "DeepSeek-Chat"
    }

    report["average_score"] = results_df["deepseek_score"].mean()
    report["min_score"] = results_df["deepseek_score"].min()
    report["max_score"] = results_df["deepseek_score"].max()
    report["average_similarity"] = results_df["deepseek_similarity"].mean()
    report["average_key_points"] = results_df["key_points_covered"].mean()

    category_scores = []
    for category in df['category'].unique():
        cat_df = results_df[results_df['category'] == category]
        if len(cat_df) > 0:
            category_scores.append({
                "category": category,
                "count": len(cat_df),
                "average_score": cat_df["deepseek_score"].mean(),
                "average_similarity": cat_df["deepseek_similarity"].mean(),
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

    # Score distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(results_df["deepseek_score"], bins=10, kde=True)
    plt.title("DeepSeek Score Distribution")
    plt.xlabel("Score (0-5)")
    plt.ylabel("Number of Questions")
    plt.axvline(report["average_score"], color='r', linestyle='--', label=f'Average: {report["average_score"]:.2f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig("deepseek_score_distribution.png")

    # Score by category
    plt.figure(figsize=(12, 6))
    sns.barplot(
        x="category",
        y="average_score",
        data=category_report_df,
        palette="viridis"
    )
    plt.title("DeepSeek Performance by Question Category")
    plt.ylabel("Average Score")
    plt.xlabel("Category")
    plt.ylim(0, 5)
    plt.tight_layout()
    plt.savefig("deepseek_performance_by_category.png")

    # Key points coverage
    key_points_data = results_df["key_points_covered"].value_counts().sort_index()
    plt.figure(figsize=(10, 6))
    key_points_data.plot(kind='bar', color='skyblue')
    plt.title("Key Information Points Coverage")
    plt.xlabel("Number of Key Points Covered")
    plt.ylabel("Number of Questions")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("deepseek_key_points_coverage.png")

    # Save summary
    with pd.ExcelWriter("deepseek_uk_immigration_test_summary.xlsx") as writer:
        report_df.to_excel(writer, sheet_name="Summary", index=False)
        category_report_df.to_excel(writer, sheet_name="Category Scores", index=False)

    # Console output
    print("\n" + "=" * 50)
    print("DeepSeek Test Summary")
    print("=" * 50)
    print(f"Total Questions: {report['total_questions']}")
    print(f"Average Score: {report['average_score']:.2f}/5")
    print(f"Min Score: {report['min_score']:.2f}/5")
    print(f"Max Score: {report['max_score']:.2f}/5")
    print(f"Average Similarity: {report['average_similarity']:.3f}")
    print(f"Average Key Points Covered: {report['average_key_points']:.2f}/3")

    print("\nPerformance by Category:")
    for cat in category_scores:
        print(
            f"  - {cat['category']}: {cat['average_score']:.2f}/5 (Similarity: {cat['average_similarity']:.3f}, Key Points: {cat['average_key_points']:.2f}/3)")

    print("\nSummary report saved to: deepseek_uk_immigration_test_summary.xlsx")
    print("Performance charts saved as:")
    print("  - deepseek_score_distribution.png")
    print("  - deepseek_performance_by_category.png")
    print("  - deepseek_key_points_coverage.png")

    return report


# Generate report
generate_summary_report(results_df)

print("\nDeepSeek test completed.")
