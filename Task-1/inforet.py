import json
import argparse
import pandas as pd
from sklearn.metrics.pairwise import (
    linear_kernel,
    cosine_similarity,
    euclidean_distances,
)

from utils import (
    create_corpus,
    get_mapping_dict,
    get_true_and_predicted,
    load_json_data,
    mean_recall_at_k,
    mean_ranking,
    mean_average_precision,
    top_k_ranks,
)

from sentence_transformers import SentenceTransformer
import torch


def main():
    parser = argparse.ArgumentParser(
        description="Patent similarity embeddings and evaluation script"
    )
    parser.add_argument(
        "--patent-section",
        type=str,
        default="title",
        help="The patent section to use (default: 'title')",
    )
    parser.add_argument(
        "--similarity-method",
        type=str,
        default="cosine_similarity",
        help="Similarity method to use: choose among 'euclidean_distances', 'cosine_similarity', or 'linear_kernel' (default: 'cosine_similarity')",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size used during embedding (default: 512)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="BAAI/bge-large-en-v1.5",
        help="Model name for SentenceTransformer (default: 'BAAI/bge-large-en-v1.5')",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="prediction1.json",
        help="Output JSON file name (default: 'prediction1.json')",
    )
    args = parser.parse_args()

    PATENT_SECTION = args.patent_section
    SIMILARITY_METHOD = args.similarity_method
    BATCH_SIZE = args.batch_size
    MODEL = args.model
    OUTPUT_JSON = args.output_json

    # Define available similarity methods and select the one requested.
    similarity_methods = {
        "euclidean_distances": euclidean_distances,
        "cosine_similarity": cosine_similarity,
        "linear_kernel": linear_kernel,
    }

    if SIMILARITY_METHOD not in similarity_methods:
        raise ValueError(
            f"Invalid similarity method: {SIMILARITY_METHOD}. Available options: {list(similarity_methods.keys())}"
        )
    get_similarity = similarity_methods[SIMILARITY_METHOD]

    # Load JSON data for citing and cited patents
    json_citing_train = load_json_data(
        "./datasets/Content_JSONs/Citing_2020_Cleaned_Content_12k/Citing_Train_Test/citing_TRAIN.json"
    )
    json_citing_test = load_json_data(
        "./datasets/Content_JSONs/Citing_2020_Cleaned_Content_12k/Citing_Train_Test/citing_TEST.json"
    )
    json_nonciting = load_json_data(
        "./datasets/Content_JSONs/Cited_2020_Uncited_2010-2019_Cleaned_Content_22k/CLEANED_CONTENT_DATASET_cited_patents_by_2020_uncited_2010-2019.json"
    )
    json_citing_to_cited = load_json_data(
        "./datasets/Citation_JSONs/Citation_Train.json"
    )

    mapping_dataset_df = pd.DataFrame(json_citing_to_cited)

    mapping_dict = get_mapping_dict(mapping_dataset_df)

    # Build corpora for citing and nonciting patents. Each corpus is a dictionary with patent id: section_text.
    citing_train = create_corpus(json_citing_train, PATENT_SECTION)
    citing_test = create_corpus(json_citing_test, PATENT_SECTION)
    nonciting = create_corpus(json_nonciting, PATENT_SECTION)

    # Initialize the model on the appropriate device.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL, device=device)

    # Create embeddings for the nonciting (cited) patents.
    nonciting_texts = [doc["text"] for doc in nonciting]
    print("First cited full text snippet:", nonciting_texts[0][:500])

    print(f'Generating "{PATENT_SECTION}" embeddings for cited patents...')
    embeddings_nonciting = model.encode(
        nonciting_texts,
        convert_to_tensor=False,
        show_progress_bar=True,
        device=device,
        batch_size=BATCH_SIZE,
    )

    # Create embeddings for citing (TRAIN) patents.
    citing_texts = [doc["text"] for doc in citing_train]
    print(f'Generating "{PATENT_SECTION}" embeddings for citing patents (TRAIN set)...')
    embeddings_citing = model.encode(
        citing_texts,
        convert_to_tensor=False,
        show_progress_bar=True,
        device=device,
        batch_size=BATCH_SIZE,
    )

    # Compute similarity scores between citing and nonciting patent embeddings.
    cosine_similarities = get_similarity(embeddings_citing, embeddings_nonciting)

    # Get top k ranks for citing patents against nonciting patents.
    k = 100  # Adjust this value if needed.
    top_k_ranktext = top_k_ranks(citing_train, nonciting, cosine_similarities, k=k)

    # Evaluate performance metrics.
    true_labels, predicted_labels, not_in_citation_mapping = get_true_and_predicted(
        mapping_dict, top_k_ranktext
    )
    mean_rank = mean_ranking(true_labels, predicted_labels)
    mean_avg_precision = mean_average_precision(true_labels, predicted_labels)
    recall_at_10 = mean_recall_at_k(true_labels, predicted_labels, k=10)
    recall_at_20 = mean_recall_at_k(true_labels, predicted_labels, k=20)
    recall_at_50 = mean_recall_at_k(true_labels, predicted_labels, k=50)
    recall_at_100 = mean_recall_at_k(true_labels, predicted_labels, k=100)

    print(f'\nNeural "{MODEL}" Embedding Approach using {PATENT_SECTION}:')
    print("Recall at 10:", round(recall_at_10, 4))
    print("Recall at 20:", round(recall_at_20, 4))
    print("Recall at 50:", round(recall_at_50, 4))
    print("Recall at 100:", round(recall_at_100, 4))
    print("Mean ranking:", round(mean_rank, 4))
    print("Mean average precision:", round(mean_avg_precision, 4))
    print("Number of patents measured:", len(predicted_labels))
    print("Number of patents not in the citation mapping:", not_in_citation_mapping)

    print()
    print(f'Generating "{PATENT_SECTION}" embeddings for citing patents (TEST set)...')
    fulltext_citing_texts_test = [doc["text"] for doc in citing_test]
    embeddings_citing_test = model.encode(
        fulltext_citing_texts_test,
        convert_to_tensor=False,
        show_progress_bar=True,
        device=device,
        batch_size=BATCH_SIZE,
    )

    fulltext_cosine_similarities_test = get_similarity(
        embeddings_citing_test, embeddings_nonciting
    )

    top_k_rank_fulltext_test = top_k_ranks(
        citing_test, nonciting, fulltext_cosine_similarities_test, k=k
    )

    with open(OUTPUT_JSON, "w") as f:
        json.dump(top_k_rank_fulltext_test, f)


if __name__ == "__main__":
    main()
