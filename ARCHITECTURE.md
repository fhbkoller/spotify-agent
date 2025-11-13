# Spotify Agent - Software Architecture

This document describes the reorganized software architecture of the Spotify Agent project, following clean architecture principles and domain-driven design.

## Project Structure

```
spotify-agent/
├── src/                          # Core application source code
│   ├── core/                     # Core application logic
│   │   ├── __init__.py
│   │   ├── agent.py              # IntelligentShuffler class
│   │   ├── models.py             # Track, SessionContext dataclasses
│   │   └── main.py               # Application entry point
│   ├── infrastructure/            # External services and persistence
│   │   ├── __init__.py
│   │   └── persistence.py         # Database and ChromaDB management
│   ├── services/                 # Business logic services
│   │   ├── __init__.py
│   │   ├── playlist_service.py   # Playlist management
│   │   ├── embedding_service.py  # Embedding generation
│   │   └── recommendation_service.py # AI recommendations
│   └── utils/                    # Utilities and helpers
│       ├── __init__.py
│       └── logging.py            # Logging configuration
├── ml/                           # Machine learning components
│   ├── __init__.py
│   ├── training/                 # Model training
│   │   ├── __init__.py
│   │   ├── train_embedding_model.py
│   │   └── train_embedding_model_v2.py
│   ├── testing/                  # Model testing
│   │   ├── __init__.py
│   │   ├── test_model.py
│   │   ├── test_model_v2.py
│   │   └── test_model_v2_training_fit.py
│   └── data/                     # Data processing
│       ├── __init__.py
│       ├── prepare_training_data.py
│       ├── convert_v1_to_v2.py
│       └── training_data_v2_augmented.py
├── scripts/                      # Utility scripts
│   ├── __init__.py
│   └── create_playlist.py
├── data/                         # Data storage
│   ├── models/                   # Trained models
│   ├── chroma_db/                # ChromaDB storage
│   ├── *.db                      # SQLite databases
│   ├── *.csv                     # Training data
│   └── *.log                     # Log files
├── tests/                        # Test files
│   └── __init__.py
├── main.py                       # Main application entry point
├── create_playlist.py            # Playlist creation script
├── requirements.txt
├── README.md
└── TECHNICAL_DOCS.md
```

## Architecture Principles

### 1. Clean Architecture
The project follows clean architecture principles with clear separation of concerns:

- **Core Layer** (`src/core/`): Contains the main business logic and domain models
- **Infrastructure Layer** (`src/infrastructure/`): Handles external dependencies like databases and APIs
- **Services Layer** (`src/services/`): Contains business logic services (currently empty, ready for future expansion)
- **Utils Layer** (`src/utils/`): Contains utility functions and configurations

### 2. Domain Separation
Different domains are clearly separated:

- **Core Application**: Main Spotify agent logic
- **Machine Learning**: All ML-related components (training, testing, data processing)
- **Scripts**: Utility scripts for specific tasks
- **Data**: All data storage and persistence

### 3. Dependency Direction
Dependencies flow inward:
- Core layer has no dependencies on infrastructure
- Infrastructure layer depends on core interfaces
- Services layer can depend on both core and infrastructure
- Utils layer is independent and can be used by any layer

## Key Components

### Core Application (`src/core/`)
- **`agent.py`**: Contains the `IntelligentShuffler` class with all the main application logic
- **`main.py`**: Application entry point with setup and main execution loop
- **`models.py`**: Data classes for `Track` and `SessionContext`

### Infrastructure (`src/infrastructure/`)
- **`persistence.py`**: Handles SQLite database and ChromaDB vector storage

### Machine Learning (`ml/`)
- **Training**: Model training scripts for different embedding models
- **Testing**: Model validation and testing scripts
- **Data**: Data preparation and processing scripts

### Scripts (`scripts/`)
- **`create_playlist.py`**: Utility to create playlists from liked songs

## Usage

### Running the Main Application
```bash
python main.py <playlist_url_or_id>
```

### Creating a Playlist
```bash
python create_playlist.py
```

### Training Models
```bash
# Prepare training data
python ml/data/prepare_training_data.py

# Convert data format
python ml/data/convert_v1_to_v2.py

# Augment data
python ml/data/training_data_v2_augmented.py

# Train model
python ml/training/train_embedding_model.py
```

### Testing Models
```bash
python ml/testing/test_model.py
python ml/testing/test_model_v2.py
```

## Benefits of This Architecture

1. **Maintainability**: Clear separation of concerns makes the code easier to understand and modify
2. **Testability**: Each layer can be tested independently
3. **Scalability**: Easy to add new features without affecting existing code
4. **Reusability**: Components can be reused across different parts of the application
5. **Dependency Management**: Clear dependency direction prevents circular dependencies

## Migration Notes

- All import statements have been updated to reflect the new structure
- File paths in configuration have been updated to point to the new `data/` directory
- Entry points have been created at the root level for backward compatibility
- All Python files compile successfully with the new structure
