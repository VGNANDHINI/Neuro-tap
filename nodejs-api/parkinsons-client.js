/**
 * Parkinson's Disease Prediction API Client for Node.js
 * Easy integration with your Node.js backend
 */

const axios = require('axios');

class ParkinsonsAPIClient {
  constructor(baseURL = 'http://localhost:5000') {
    this.baseURL = baseURL;
    this.client = axios.create({
      baseURL: this.baseURL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json'
      }
    });
  }

  /**
   * Check API health
   */
  async healthCheck() {
    try {
      const response = await this.client.get('/health');
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  /**
   * Predict using Spiral Drawing Test
   */
  async predictSpiral(features) {
    try {
      const response = await this.client.post('/predict/spiral', { features });
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  /**
   * Predict using Tapping Test
   */
  async predictTapping(features) {
    try {
      const response = await this.client.post('/predict/tapping', { features });
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  /**
   * Predict using Reaction Test
   */
  async predictReaction(features) {
    try {
      const response = await this.client.post('/predict/reaction', { features });
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  /**
   * Predict using Voice Test
   */
  async predictVoice(features) {
    try {
      const response = await this.client.post('/predict/voice', { features });
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  /**
   * Predict using Non-Motor Test
   */
  async predictNonMotor(features) {
    try {
      const response = await this.client.post('/predict/nonmotor', { features });
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  /**
   * Predict using Cognitive Test
   */
  async predictCognitive(features) {
    try {
      const response = await this.client.post('/predict/cognitive', { features });
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  /**
   * Predict using Ensemble Model (Final Risk Assessment)
   */
  async predictEnsemble(features) {
    try {
      const response = await this.client.post('/predict/ensemble', { features });
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  /**
   * Comprehensive prediction using all available test data
   */
  async predictComprehensive(testData) {
    try {
      const response = await this.client.post('/predict/comprehensive', testData);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  /**
   * Get required features for a specific model
   */
  async getModelFeatures(modelName) {
    try {
      const response = await this.client.get(`/features/${modelName}`);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  /**
   * Error handler
   */
  handleError(error) {
    if (error.response) {
      return new Error(`API Error: ${error.response.data.error || error.response.statusText}`);
    } else if (error.request) {
      return new Error('Network Error: No response from API server');
    } else {
      return new Error(`Error: ${error.message}`);
    }
  }
}

module.exports = ParkinsonsAPIClient;

// Example usage
if (require.main === module) {
  (async () => {
    const client = new ParkinsonsAPIClient('http://localhost:5000');
    
    try {
      // Health check
      console.log('Checking API health...');
      const health = await client.healthCheck();
      console.log('Health:', health);

      // Example: Get features for spiral model
      console.log('\nGetting spiral model features...');
      const features = await client.getModelFeatures('spiral_svm');
      console.log('Features:', features);

      // Example prediction (you need to provide actual feature values)
      // const spiralResult = await client.predictSpiral({
      //   feature1: 0.5,
      //   feature2: 0.3,
      //   // ... add all required features
      // });
      // console.log('Prediction:', spiralResult);

    } catch (error) {
      console.error('Error:', error.message);
    }
  })();
}