# Machine Learning & Data Science Course

This repository contains a comprehensive series of Machine Learning and Data Science sprints, covering fundamental concepts from data visualization to advanced neural networks and generative AI.

## 📚 Course Overview

The course is structured as a series of progressive sprints (ML9-ML17.2), each focusing on specific aspects of machine learning and data science. Each sprint includes practical implementations, comprehensive testing, and real-world applications.

## 🗂️ Project Structure

```
ML-Course/
├── ML9/          # [Content not visible in current documents]
├── ML10/         # [Content not visible in current documents]
├── ML11/         # Data Visualization and Exploration
├── ML12/         # Free Text and NLP in Data Science
├── ML13/         # Supervised Learning - Regression
├── ML14/         # [Content not visible in current documents]
├── ML15/         # Unsupervised Learning - Clustering and Dimensionality Reduction
├── ML16/         # Introduction to Neural Networks
├── ML17/         # Generative AI in Data Science 
└── ML17.2/       # Generative AI in Data Science #2
```

## 📋 Sprint Details

### ML11 - Data Visualization and Exploration
**Focus**: Foundation of data visualization techniques and exploratory data analysis

**Key Topics**:
- Importance of data visualization in data science
- Key principles of effective data visualization (clarity, simplicity, accuracy)
- Plotting different data types (1D, 2D, 3D) with Matplotlib
- Exploratory Data Analysis (EDA) techniques
- Interactive visualizations with Plotly and Bokeh

**Technologies**: Matplotlib, Seaborn, Plotly, Bokeh, NumPy, Pandas

**Tasks**:
1. Bar chart creation for categorical data frequency
2. Scatter plots with Seaborn following visualization principles
3. Multi-dimensional data plotting (1D line, 2D scatter, 3D scatter)
4. Comprehensive EDA with descriptive statistics, outlier detection, and correlation analysis
5. Interactive plot creation with Plotly

### ML12 - Free Text and NLP in Data Science
**Focus**: Natural Language Processing fundamentals and text analysis

**Key Topics**:
- Text preprocessing and tokenization
- Stemming and lemmatization techniques
- Bag of Words (BoW) representation
- TF-IDF calculation and analysis
- Word2Vec model training and word similarity
- N-gram generation and frequency analysis

**Technologies**: NLTK, scikit-learn, Gensim, Pandas

**Tasks**:
1. Text cleaning and tokenization implementation
2. Stemming and lemmatization comparison
3. Bag of Words model creation
4. TF-IDF calculation and representation
5. Word2Vec training for text representation
6. N-gram generation and frequency counting

### ML13 - Supervised Learning - Regression
**Focus**: Regression algorithms and model evaluation

**Key Topics**:
- Linear regression implementation and evaluation
- Polynomial regression for non-linear relationships
- Model evaluation metrics (MAE, MSE, RMSE)
- Data visualization for regression analysis
- CSV data handling and preprocessing

**Technologies**: scikit-learn, Pandas, Matplotlib, NumPy

**Tasks**:
1. Linear regression with comprehensive model evaluation
2. Polynomial regression for house price prediction

**Datasets**:
- `data.csv`: Feature-target pairs for linear regression
- `house_data.csv`: House size and price data for polynomial regression

### ML15 - Unsupervised Learning - Clustering and Dimensionality Reduction
**Focus**: Clustering algorithms and anomaly detection

**Key Topics**:
- K-Means clustering implementation from scratch
- Customer segmentation applications
- DBSCAN algorithm for anomaly detection
- Density-based clustering concepts
- Cluster visualization and evaluation

**Technologies**: NumPy, Matplotlib, scikit-learn concepts

**Tasks**:
1. Customer segmentation using K-Means clustering
2. Anomaly detection using DBSCAN algorithm

### ML16 - Introduction to Neural Networks
**Focus**: Deep learning fundamentals with TensorFlow/Keras

**Key Topics**:
- Neural network architecture design
- Function approximation with neural networks
- Regression with deep learning
- Model training and evaluation
- Real-world applications (house price prediction)

**Technologies**: TensorFlow, Keras, scikit-learn, Matplotlib

**Tasks**:
1. Function value forecasting (`f(x) = sin(x) + 0.1x²`)
2. House price prediction using California housing dataset

### ML17.2 - Generative AI in Data Science
**Focus**: Advanced generative models and data compression

**Key Topics**:
- Variational Autoencoders (VAE) implementation
- Data compression and reconstruction
- Latent space representation
- Generative model training with PyTorch
- Wine dataset analysis

**Technologies**: PyTorch, scikit-learn, NumPy

**Tasks**:
1. VAE implementation for data compression and reconstruction
2. Wine dataset dimensionality reduction

## 🛠️ Technical Requirements

### Dependencies
Each sprint includes a `requirements.txt` file with specific version requirements. Common dependencies include:

- **Data Processing**: pandas, numpy, scikit-learn
- **Visualization**: matplotlib, seaborn, plotly, bokeh
- **NLP**: nltk, gensim
- **Deep Learning**: tensorflow, keras, torch
- **Testing**: pytest

### Development Environment
- **Python Version**: 3.10
- **Virtual Environment**: Each project uses venv for dependency isolation
- **Testing Framework**: pytest for unit testing
- **CI/CD**: GitHub Actions for automated testing

## 🧪 Testing Strategy

Each sprint includes comprehensive test suites:

- **Unit Tests**: Validate individual function correctness
- **Integration Tests**: Ensure components work together
- **Model Validation**: Verify machine learning model outputs
- **Data Validation**: Check data loading and preprocessing

### Running Tests
```bash
# Navigate to any sprint directory
cd ML[XX]/

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest
```

## 📊 Key Learning Outcomes

### Data Science Fundamentals
- Data preprocessing and cleaning techniques
- Exploratory data analysis best practices
- Statistical analysis and visualization

### Machine Learning Algorithms
- Supervised learning (regression, classification concepts)
- Unsupervised learning (clustering, dimensionality reduction)
- Model evaluation and validation techniques

### Advanced Topics
- Neural network architecture design
- Deep learning for regression and function approximation
- Generative models and autoencoders
- Natural language processing fundamentals

### Software Engineering Practices
- Test-driven development
- Code organization and modularity
- Version control and CI/CD integration
- Documentation and reproducibility

## 🚀 Getting Started

1. **Clone the Repository**
   ```bash
   git clone [repository-url]
   cd ML-Course
   ```

2. **Choose a Sprint**
   ```bash
   cd ML[XX]/  # Replace XX with desired sprint number
   ```

3. **Set Up Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

4. **Run the Code**
   ```bash
   python src/[task_file].py
   ```

5. **Run Tests**
   ```bash
   pytest
   ```

## 📈 Progression Path

The sprints are designed to build upon each other:

1. **Foundation** (ML11-ML12): Data handling, visualization, and text processing
2. **Classical ML** (ML13, ML15): Regression and clustering algorithms
3. **Deep Learning** (ML16): Neural networks and advanced modeling
4. **Advanced AI** (ML17.2): Generative models and modern AI techniques

## 🔧 Development Notes

### Important Guidelines
- **Plot Management**: Always include `plt.close()` after `plt.show()` in plotting functions to prevent memory issues in CI/CD environments
- **Testing**: Each implementation includes corresponding unit tests
- **Data Handling**: Robust error handling for empty datasets and missing files
- **Code Quality**: Consistent formatting and comprehensive documentation

### Common Patterns
- CSV data loading with validation
- Model training with proper train/test splits
- Visualization with clear labels and titles
- Comprehensive error handling and edge case testing

## 📝 Contributing

When working on assignments:

1. Follow the existing code structure and naming conventions
2. Include comprehensive docstrings for all functions
3. Write tests for new functionality
4. Ensure all plots include proper `plt.close()` calls
5. Validate data loading and preprocessing steps

## 🎯 Assessment Criteria

Each sprint is evaluated based on:

- **Functionality**: Correct implementation of required algorithms
- **Code Quality**: Clean, well-documented, and maintainable code
- **Testing**: Comprehensive test coverage and passing test suites
- **Visualization**: Clear and informative data visualizations
- **Analysis**: Proper interpretation of results and model evaluation

## 📞 Support

For technical issues or questions:
- Check individual sprint README files for specific requirements
- Review test files for expected functionality
- Ensure all dependencies are properly installed
- Verify Python version compatibility (3.10 recommended)

---

*This course provides a comprehensive foundation in machine learning and data science, progressing from basic data visualization to advanced generative AI techniques. Each sprint builds practical skills while maintaining high code quality standards through testing and documentation.*