import os
import json
import pickle
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
from loguru import logger

from config.models_config import AutocorrectModelConfig, ModelType
from core.phonetic_hash import phonetic_hash
from core.trie import Trie
from core.distance import weighted_edit_distance


def load_base_words_from_file(filepath: str) -> List[str]:
    """Loads words from the specified dictionary file."""
    words = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip().lower()
            if word:
                words.append(word)
    return list(set(words))  # Ensure uniqueness


def build_trie_and_phonetic_index(base_words: List[str]) -> Tuple[Trie, Dict[str, List[str]]]:
    """Build trie and phonetic index from word list."""
    all_words = list(set(base_words))  # Ensure uniqueness
    
    trie = Trie()
    phonetic_index = {}
    for word in all_words:
        trie.insert(word)
        key = phonetic_hash(word)
        phonetic_index.setdefault(key, []).append(word)
    return trie, phonetic_index


class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False


class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word: str):
        current = self.root
        for char in word:
            if char not in current.children:
                current.children[char] = TrieNode()
            current = current.children[char]
        current.is_end_of_word = True
    
    def search(self, word: str) -> bool:
        current = self.root
        for char in word:
            if char not in current.children:
                return False
            current = current.children[char]
        return current.is_end_of_word


class AutocorrectService:
    """Service for managing autocorrect models and performing spell checking operations"""

    # Constants
    MAX_SUGGESTIONS = 5
    EDIT_DISTANCE_THRESHOLD = 3

    def __init__(self):
        self._models: Dict[str, Dict[str, Any]] = {}
        self._model_configs: Dict[str, AutocorrectModelConfig] = {}

    def load_model(self, model_config: AutocorrectModelConfig) -> bool:
        """Load an autocorrect model into memory"""
        try:
            logger.info(f"Loading autocorrect model: {model_config.name}")

            # Validate model files exist
            if not os.path.exists(model_config.dictionary_path):
                logger.error(f"Dictionary file not found: {model_config.dictionary_path}")
                return False

            if model_config.phonetic_index_path and not os.path.exists(model_config.phonetic_index_path):
                logger.error(f"Phonetic index file not found: {model_config.phonetic_index_path}")
                return False

            # Load dictionary using utility function
            word_list = load_base_words_from_file(model_config.dictionary_path)

            # Load or build phonetic index and trie
            if model_config.phonetic_index_path and os.path.exists(model_config.phonetic_index_path):
                # Load pre-built phonetic index
                with open(model_config.phonetic_index_path, 'rb') as f:
                    phonetic_index = pickle.load(f)
                logger.info(f"Loaded pre-built phonetic index with {len(phonetic_index)} entries")
                
                # Build trie from word list
                trie = Trie()
                for word in word_list:
                    trie.insert(word)
            else:
                # Build both trie and phonetic index using utility function
                logger.info("Building trie and phonetic index from dictionary...")
                trie, phonetic_index = build_trie_and_phonetic_index(word_list)
                
                # Save phonetic index if path is provided
                if model_config.phonetic_index_path:
                    os.makedirs(os.path.dirname(model_config.phonetic_index_path), exist_ok=True)
                    with open(model_config.phonetic_index_path, 'wb') as f:
                        pickle.dump(phonetic_index, f)
                    logger.info(f"Saved phonetic index to {model_config.phonetic_index_path}")

            # Create fallback pool
            fallback_pool = list(set(word for words in phonetic_index.values() for word in words))

            # Store the model components
            self._models[model_config.name] = {
                'trie': trie,
                'phonetic_index': phonetic_index,
                'fallback_pool': fallback_pool,
                'word_count': len(word_list)
            }
            self._model_configs[model_config.name] = model_config

            logger.info(f"Successfully loaded autocorrect model: {model_config.name} "
                       f"({len(word_list)} words, {len(phonetic_index)} phonetic keys)")
            return True

        except Exception as e:
            logger.error(f"Failed to load autocorrect model {model_config.name}: {str(e)}")
            return False

    def unload_model(self, model_name: str) -> bool:
        """Unload an autocorrect model from memory"""
        try:
            if model_name in self._models:
                del self._models[model_name]
                del self._model_configs[model_name]
                logger.info(f"Unloaded autocorrect model: {model_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to unload autocorrect model {model_name}: {str(e)}")
            return False

    def get_loaded_models(self) -> Dict[str, AutocorrectModelConfig]:
        """Get all currently loaded models"""
        return self._model_configs.copy()

    def is_model_loaded(self, model_name: str) -> bool:
        """Check if a model is currently loaded"""
        return model_name in self._models

    def phonetic_hash(self, word: str) -> str:
        """Generate phonetic hash for a word - using imported function"""
        return phonetic_hash(word)

    def weighted_edit_distance(self, word1: str, word2: str) -> float:
        """Calculate weighted edit distance between two words - using imported function"""
        return weighted_edit_distance(word1, word2)

    def suggest_corrections(
        self, 
        word: str, 
        model_name: str,
        max_suggestions: Optional[int] = None
    ) -> List[str]:
        """Suggest corrections for a misspelled word"""
        try:
            if model_name not in self._models:
                logger.error(f"Model {model_name} is not loaded")
                return []

            model = self._models[model_name]
            phonetic_index = model['phonetic_index']
            fallback_pool = model['fallback_pool']
            
            if max_suggestions is None:
                max_suggestions = self.MAX_SUGGESTIONS

            word = word.lower()
            phonetic_key = phonetic_hash(word)
            candidates = phonetic_index.get(phonetic_key, [])
            
            use_fallback = len(candidates) == 0
            if use_fallback:
                logger.debug(f"[{word}] No phonetic match found. Using fallback approach.")
                candidates = fallback_pool
            else:
                logger.debug(f"[{word}] Using phonetic hash match: {phonetic_key} ({len(candidates)} candidates)")

            scored = []
            for candidate in candidates:
                dist = weighted_edit_distance(word, candidate)
                if dist <= self.EDIT_DISTANCE_THRESHOLD:
                    scored.append((candidate, dist))

            # Sort by distance, then alphabetically
            ranked = sorted(scored, key=lambda x: (x[1], x[0]))
            top_suggestions = [word for word, _ in ranked[:max_suggestions]]
            
            logger.debug(f"[{word}] Suggestions: {top_suggestions}")
            return top_suggestions

        except Exception as e:
            logger.error(f"Failed to suggest corrections for '{word}' with model {model_name}: {str(e)}")
            return []

    def check_spelling(
        self,
        word: str,
        model_name: str
    ) -> bool:
        """Check if a word is spelled correctly"""
        try:
            if model_name not in self._models:
                logger.error(f"Model {model_name} is not loaded")
                return False

            model = self._models[model_name]
            trie = model['trie']
            
            word_clean = word.lower().strip(",.?!;:'\"")
            return trie.search(word_clean)

        except Exception as e:
            logger.error(f"Failed to check spelling for '{word}' with model {model_name}: {str(e)}")
            return False

    def spellcheck_sentence(
        self,
        sentence: str,
        model_name: str,
        max_suggestions: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform spell checking on a sentence
        
        Args:
            sentence: Input sentence to check
            model_name: Name of the autocorrect model to use
            max_suggestions: Maximum number of suggestions per word
            
        Returns:
            List of dictionaries with word analysis results
        """
        try:
            if model_name not in self._models:
                logger.error(f"Model {model_name} is not loaded")
                return []

            model = self._models[model_name]
            trie = model['trie']
            
            if max_suggestions is None:
                max_suggestions = self.MAX_SUGGESTIONS

            results = []
            tokens = sentence.strip().split()

            for token in tokens:
                token_clean = token.lower().strip(",.?!;:'\"")
                
                if trie.search(token_clean):
                    results.append({
                        "word": token,
                        "correct": True,
                        "suggestions": []
                    })
                else:
                    suggestions = self.suggest_corrections(token_clean, model_name, max_suggestions)
                    results.append({
                        "word": token,
                        "correct": False,
                        "suggestions": suggestions
                    })

            return results

        except Exception as e:
            logger.error(f"Failed to spellcheck sentence with model {model_name}: {str(e)}")
            return []

    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a loaded model"""
        if model_name not in self._models:
            return None

        config = self._model_configs[model_name]
        model = self._models[model_name]
        
        return {
            'name': config.name,
            'language': config.language,
            'description': config.description,
            'word_count': model['word_count'],
            'phonetic_keys': len(model['phonetic_index']),
            'is_active': config.is_active,
            'dictionary_path': config.dictionary_path,
            'phonetic_index_path': config.phonetic_index_path
        }

    def export_model_stats(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Export detailed statistics about a model"""
        if model_name not in self._models:
            return None

        model = self._models[model_name]
        phonetic_index = model['phonetic_index']
        
        # Calculate statistics
        phonetic_distribution = {k: len(v) for k, v in phonetic_index.items()}
        avg_words_per_key = sum(phonetic_distribution.values()) / len(phonetic_distribution)
        
        return {
            'total_words': model['word_count'],
            'phonetic_keys': len(phonetic_index),
            'avg_words_per_phonetic_key': avg_words_per_key,
            'phonetic_distribution': phonetic_distribution,
            'most_common_phonetic_keys': sorted(
                phonetic_distribution.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
        }