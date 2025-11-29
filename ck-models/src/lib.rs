use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub name: String,
    pub provider: String,
    pub dimensions: usize,
    pub max_tokens: usize,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRegistry {
    pub models: HashMap<String, ModelConfig>,
    pub default_model: String,
}

impl Default for ModelRegistry {
    fn default() -> Self {
        let mut models = HashMap::new();

        models.insert(
            "bge-small".to_string(),
            ModelConfig {
                name: "BAAI/bge-small-en-v1.5".to_string(),
                provider: "fastembed".to_string(),
                dimensions: 384,
                max_tokens: 512,
                description: "Small, fast English embedding model".to_string(),
            },
        );

        models.insert(
            "minilm".to_string(),
            ModelConfig {
                name: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
                provider: "fastembed".to_string(),
                dimensions: 384,
                max_tokens: 256,
                description: "Lightweight English embedding model".to_string(),
            },
        );

        // Add enhanced models
        models.insert(
            "nomic-v1.5".to_string(),
            ModelConfig {
                name: "nomic-embed-text-v1.5".to_string(),
                provider: "fastembed".to_string(),
                dimensions: 768,
                max_tokens: 8192,
                description: "High-quality English embedding model with large context window"
                    .to_string(),
            },
        );

        models.insert(
            "jina-code".to_string(),
            ModelConfig {
                name: "jina-embeddings-v2-base-code".to_string(),
                provider: "fastembed".to_string(),
                dimensions: 768,
                max_tokens: 8192,
                description: "Code-specific embedding model optimized for programming tasks"
                    .to_string(),
            },
        );

        // Multilingual models
        models.insert(
            "multilingual-e5-small".to_string(),
            ModelConfig {
                name: "intfloat/multilingual-e5-small".to_string(),
                provider: "fastembed".to_string(),
                dimensions: 384,
                max_tokens: 512,
                description: "Fast multilingual embedding model (100+ languages)".to_string(),
            },
        );

        models.insert(
            "multilingual-e5-base".to_string(),
            ModelConfig {
                name: "intfloat/multilingual-e5-base".to_string(),
                provider: "fastembed".to_string(),
                dimensions: 768,
                max_tokens: 512,
                description: "Balanced multilingual embedding model (100+ languages, recommended)"
                    .to_string(),
            },
        );

        models.insert(
            "multilingual-e5-large".to_string(),
            ModelConfig {
                name: "intfloat/multilingual-e5-large".to_string(),
                provider: "fastembed".to_string(),
                dimensions: 1024,
                max_tokens: 512,
                description: "High-quality multilingual embedding model (100+ languages)"
                    .to_string(),
            },
        );

        models.insert(
            "paraphrase-multilingual-mpnet".to_string(),
            ModelConfig {
                name: "sentence-transformers/paraphrase-multilingual-mpnet-base-v2".to_string(),
                provider: "fastembed".to_string(),
                dimensions: 768,
                max_tokens: 128,
                description: "Multilingual paraphrase model with strong semantic understanding (50+ languages)".to_string(),
            },
        );

        // Modern architecture
        models.insert(
            "modernbert-large".to_string(),
            ModelConfig {
                name: "lightonai/modernbert-embed-large".to_string(),
                provider: "fastembed".to_string(),
                dimensions: 1024,
                max_tokens: 8192,
                description:
                    "Modern BERT architecture with 8K context window and improved performance"
                        .to_string(),
            },
        );

        Self {
            models,
            default_model: "bge-small".to_string(), // Keep BGE as default for backward compatibility
        }
    }
}

impl ModelRegistry {
    pub fn load(path: &Path) -> Result<Self> {
        if path.exists() {
            let data = std::fs::read_to_string(path)?;
            Ok(serde_json::from_str(&data)?)
        } else {
            Ok(Self::default())
        }
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let data = serde_json::to_string_pretty(self)?;
        std::fs::write(path, data)?;
        Ok(())
    }

    pub fn get_model(&self, name: &str) -> Option<&ModelConfig> {
        self.models.get(name)
    }

    pub fn get_default_model(&self) -> Option<&ModelConfig> {
        self.models.get(&self.default_model)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectConfig {
    pub model: String,
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    pub index_backend: String,
}

impl Default for ProjectConfig {
    fn default() -> Self {
        Self {
            model: "bge-small".to_string(),
            chunk_size: 512,
            chunk_overlap: 128,
            index_backend: "hnsw".to_string(),
        }
    }
}

impl ProjectConfig {
    pub fn load(path: &Path) -> Result<Self> {
        if path.exists() {
            let data = std::fs::read_to_string(path)?;
            Ok(serde_json::from_str(&data)?)
        } else {
            Ok(Self::default())
        }
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let data = serde_json::to_string_pretty(self)?;
        std::fs::write(path, data)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_registry_default() {
        let registry = ModelRegistry::default();

        // Verify default model is set correctly
        assert_eq!(registry.default_model, "bge-small");
        assert!(registry.get_default_model().is_some());
    }

    #[test]
    fn test_all_models_registered() {
        let registry = ModelRegistry::default();

        // All expected model short names
        let expected_models = [
            "bge-small",
            "minilm",
            "nomic-v1.5",
            "jina-code",
            "multilingual-e5-small",
            "multilingual-e5-base",
            "multilingual-e5-large",
            "paraphrase-multilingual-mpnet",
            "modernbert-large",
        ];

        for model_name in expected_models {
            assert!(
                registry.get_model(model_name).is_some(),
                "Model '{}' should be registered",
                model_name
            );
        }

        // Verify total count
        assert_eq!(
            registry.models.len(),
            expected_models.len(),
            "Expected {} models, found {}",
            expected_models.len(),
            registry.models.len()
        );
    }

    #[test]
    fn test_multilingual_e5_small_config() {
        let registry = ModelRegistry::default();
        let model = registry
            .get_model("multilingual-e5-small")
            .expect("multilingual-e5-small should exist");

        assert_eq!(model.name, "intfloat/multilingual-e5-small");
        assert_eq!(model.provider, "fastembed");
        assert_eq!(model.dimensions, 384);
        assert_eq!(model.max_tokens, 512);
        assert!(model.description.contains("multilingual"));
    }

    #[test]
    fn test_multilingual_e5_base_config() {
        let registry = ModelRegistry::default();
        let model = registry
            .get_model("multilingual-e5-base")
            .expect("multilingual-e5-base should exist");

        assert_eq!(model.name, "intfloat/multilingual-e5-base");
        assert_eq!(model.provider, "fastembed");
        assert_eq!(model.dimensions, 768);
        assert_eq!(model.max_tokens, 512);
        assert!(model.description.contains("multilingual"));
        assert!(model.description.contains("recommended"));
    }

    #[test]
    fn test_multilingual_e5_large_config() {
        let registry = ModelRegistry::default();
        let model = registry
            .get_model("multilingual-e5-large")
            .expect("multilingual-e5-large should exist");

        assert_eq!(model.name, "intfloat/multilingual-e5-large");
        assert_eq!(model.provider, "fastembed");
        assert_eq!(model.dimensions, 1024);
        assert_eq!(model.max_tokens, 512);
        assert!(model.description.contains("multilingual"));
    }

    #[test]
    fn test_paraphrase_multilingual_mpnet_config() {
        let registry = ModelRegistry::default();
        let model = registry
            .get_model("paraphrase-multilingual-mpnet")
            .expect("paraphrase-multilingual-mpnet should exist");

        assert_eq!(
            model.name,
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        );
        assert_eq!(model.provider, "fastembed");
        assert_eq!(model.dimensions, 768);
        assert_eq!(model.max_tokens, 128); // Only 128 token max per HuggingFace docs
        assert!(model.description.to_lowercase().contains("multilingual"));
    }

    #[test]
    fn test_modernbert_large_config() {
        let registry = ModelRegistry::default();
        let model = registry
            .get_model("modernbert-large")
            .expect("modernbert-large should exist");

        assert_eq!(model.name, "lightonai/modernbert-embed-large");
        assert_eq!(model.provider, "fastembed");
        assert_eq!(model.dimensions, 1024);
        assert_eq!(model.max_tokens, 8192); // 8K context window per HuggingFace docs
        assert!(model.description.to_lowercase().contains("modern"));
        assert!(model.description.contains("8K")); // Verify description mentions context window
    }

    #[test]
    fn test_model_dimensions_consistency() {
        let registry = ModelRegistry::default();

        // Small models should have 384 dimensions
        let small_models = ["bge-small", "minilm", "multilingual-e5-small"];
        for name in small_models {
            let model = registry.get_model(name).expect(name);
            assert_eq!(
                model.dimensions, 384,
                "Model '{}' should have 384 dimensions",
                name
            );
        }

        // Medium models should have 768 dimensions
        let medium_models = [
            "nomic-v1.5",
            "jina-code",
            "multilingual-e5-base",
            "paraphrase-multilingual-mpnet",
        ];
        for name in medium_models {
            let model = registry.get_model(name).expect(name);
            assert_eq!(
                model.dimensions, 768,
                "Model '{}' should have 768 dimensions",
                name
            );
        }

        // Large models should have 1024 dimensions
        let large_models = ["multilingual-e5-large", "modernbert-large"];
        for name in large_models {
            let model = registry.get_model(name).expect(name);
            assert_eq!(
                model.dimensions, 1024,
                "Model '{}' should have 1024 dimensions",
                name
            );
        }
    }

    #[test]
    fn test_all_models_have_fastembed_provider() {
        let registry = ModelRegistry::default();

        for (name, config) in &registry.models {
            assert_eq!(
                config.provider, "fastembed",
                "Model '{}' should have fastembed provider",
                name
            );
        }
    }

    #[test]
    fn test_model_max_tokens_reasonable() {
        let registry = ModelRegistry::default();

        for (name, config) in &registry.models {
            // All models should have max_tokens of at least 64 (minimum practical limit)
            // Note: paraphrase-multilingual-mpnet has only 128 tokens per HuggingFace docs
            assert!(
                config.max_tokens >= 64,
                "Model '{}' max_tokens ({}) should be at least 64",
                name,
                config.max_tokens
            );

            // No model should exceed 8192 tokens
            assert!(
                config.max_tokens <= 8192,
                "Model '{}' max_tokens ({}) should not exceed 8192",
                name,
                config.max_tokens
            );
        }
    }

    #[test]
    fn test_model_names_are_valid_fastembed_names() {
        let registry = ModelRegistry::default();

        // These are the full model names that should be recognized by fastembed
        let valid_fastembed_names = [
            "BAAI/bge-small-en-v1.5",
            "sentence-transformers/all-MiniLM-L6-v2",
            "nomic-embed-text-v1.5",
            "jina-embeddings-v2-base-code",
            "intfloat/multilingual-e5-small",
            "intfloat/multilingual-e5-base",
            "intfloat/multilingual-e5-large",
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            "lightonai/modernbert-embed-large",
        ];

        // Verify all registered models use valid fastembed names
        for (short_name, config) in &registry.models {
            assert!(
                valid_fastembed_names.contains(&config.name.as_str()),
                "Model '{}' has unrecognized fastembed name: '{}'",
                short_name,
                config.name
            );
        }
    }

    #[test]
    fn test_registry_serialization_roundtrip() {
        let registry = ModelRegistry::default();

        // Serialize to JSON
        let json = serde_json::to_string(&registry).expect("Should serialize");

        // Deserialize back
        let restored: ModelRegistry = serde_json::from_str(&json).expect("Should deserialize");

        // Verify all models survived the roundtrip
        assert_eq!(registry.models.len(), restored.models.len());
        assert_eq!(registry.default_model, restored.default_model);

        for (name, config) in &registry.models {
            let restored_config = restored.get_model(name).expect("Model should exist");
            assert_eq!(config.name, restored_config.name);
            assert_eq!(config.dimensions, restored_config.dimensions);
            assert_eq!(config.max_tokens, restored_config.max_tokens);
        }
    }

    #[test]
    fn test_project_config_default() {
        let config = ProjectConfig::default();

        assert_eq!(config.model, "bge-small");
        assert_eq!(config.chunk_size, 512);
        assert_eq!(config.chunk_overlap, 128);
        assert_eq!(config.index_backend, "hnsw");
    }
}
