import json
import logging
from collections import Counter
import nlpaug.augmenter.word as naw
from tqdm import tqdm
import random
import nltk

try:
    nltk.data.find('wordnet')
    nltk.download('averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger_eng')
    nltk.download('omw-1.4')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PromiseAugmenter:
    def __init__(self):
        # Initialize word-level augmenters
        self.synonym_aug = naw.SynonymAug(aug_src='wordnet')
        self.random_aug = naw.RandomWordAug(action="swap")

    def augment_text(self, text, techniques=None):
        """Apply multiple augmentation techniques to a text"""
        if techniques is None:
            techniques = ['synonym', 'random']

        augmented_texts = []

        try:
            if 'synonym' in techniques:
                # Replace some words with synonyms
                aug_text = self.synonym_aug.augment(text)[0]
                augmented_texts.append(aug_text)

            if 'random' in techniques:
                # Randomly swap some words
                aug_text = self.random_aug.augment(text)[0]
                augmented_texts.append(aug_text)

        except Exception as e:
            logger.warning(f"Augmentation error for text: {str(e)}")
            return []

        return augmented_texts


def balance_dataset(json_file, output_file):
    """Balance the dataset using augmentation"""
    # Load and analyze original data
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Count original distribution
    promise_counts = Counter(item['promise_status'] for item in data)
    majority_class = max(promise_counts.items(), key=lambda x: x[1])[0]
    majority_count = promise_counts[majority_class]

    logger.info("Original distribution:")
    for status, count in promise_counts.items():
        logger.info(f"{status}: {count}")

    # Initialize augmenter
    augmenter = PromiseAugmenter()

    # Augment minority class
    augmented_data = data.copy()
    minority_class = 'No' if majority_class == 'Yes' else 'Yes'

    # Get all examples of minority class
    minority_examples = [item for item in data if item['promise_status'] == minority_class]
    needed_samples = majority_count - len(minority_examples)

    logger.info(f"Generating {needed_samples} samples for class {minority_class}")

    # Generate augmented samples
    new_samples = []
    while len(new_samples) < needed_samples:
        original = random.choice(minority_examples)
        augmented_texts = augmenter.augment_text(original['data'])

        for aug_text in augmented_texts:
            if len(new_samples) < needed_samples:
                new_sample = original.copy()
                new_sample['data'] = aug_text
                new_samples.append(new_sample)

                # Log example augmentation
                if len(new_samples) == 1:
                    logger.info("\nExample augmentation:")
                    logger.info(f"Original: {original['data'][:100]}...")
                    logger.info(f"Augmented: {aug_text[:100]}...")

    augmented_data.extend(new_samples)

    # Verify final distribution
    final_counts = Counter(item['promise_status'] for item in augmented_data)
    logger.info("\nFinal distribution:")
    for status, count in final_counts.items():
        logger.info(f"{status}: {count}")

    # Save augmented dataset
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(augmented_data, f, indent=2)

    logger.info(f"\nAugmented dataset saved to {output_file}")
    logger.info(f"Final dataset size: {len(augmented_data)}")

    return augmented_data


if __name__ == "__main__":
    input_file = "../../data/PromiseEval_Trainset_English.json"
    output_file = "../../data/english_train_augmented.json"

    # Balance dataset
    balanced_data = balance_dataset(input_file, output_file)