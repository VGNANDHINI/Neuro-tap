/**
 * Express.js Server with Parkinson's API Integration
 * Example implementation showing how to integrate the ML API with your Node.js backend
 */

const express = require('express');
const cors = require('cors');
const ParkinsonsAPIClient = require('./parkinsons-client');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

// Initialize Parkinson's API client
const parkinsonsAPI = new ParkinsonsAPIClient('http://localhost:5000');

// Health check endpoint
app.get('/api/health', async (req, res) => {
  try {
    const health = await parkinsonsAPI.healthCheck();
    res.json({ status: 'ok', ml_api: health });
  } catch (error) {
    res.status(500).json({ status: 'error', message: error.message });
  }
});

// Spiral test endpoint
app.post('/api/parkinsons/spiral', async (req, res) => {
  try {
    const { features } = req.body;
    
    if (!features) {
      return res.status(400).json({ error: 'Features are required' });
    }
    
    const result = await parkinsonsAPI.predictSpiral(features);
    
    // You can add additional processing, logging, or database storage here
    // await saveToDatabase(req.user.id, 'spiral', result);
    
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Tapping test endpoint
app.post('/api/parkinsons/tapping', async (req, res) => {
  try {
    const { features } = req.body;
    
    if (!features) {
      return res.status(400).json({ error: 'Features are required' });
    }
    
    const result = await parkinsonsAPI.predictTapping(features);
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Reaction test endpoint
app.post('/api/parkinsons/reaction', async (req, res) => {
  try {
    const { features } = req.body;
    
    if (!features) {
      return res.status(400).json({ error: 'Features are required' });
    }
    
    const result = await parkinsonsAPI.predictReaction(features);
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Voice test endpoint
app.post('/api/parkinsons/voice', async (req, res) => {
  try {
    const { features } = req.body;
    
    if (!features) {
      return res.status(400).json({ error: 'Features are required' });
    }
    
    const result = await parkinsonsAPI.predictVoice(features);
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Non-motor test endpoint
app.post('/api/parkinsons/nonmotor', async (req, res) => {
  try {
    const { features } = req.body;
    
    if (!features) {
      return res.status(400).json({ error: 'Features are required' });
    }
    
    const result = await parkinsonsAPI.predictNonMotor(features);
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Cognitive test endpoint
app.post('/api/parkinsons/cognitive', async (req, res) => {
  try {
    const { features } = req.body;
    
    if (!features) {
      return res.status(400).json({ error: 'Features are required' });
    }
    
    const result = await parkinsonsAPI.predictCognitive(features);
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Ensemble prediction endpoint
app.post('/api/parkinsons/ensemble', async (req, res) => {
  try {
    const { features } = req.body;
    
    if (!features) {
      return res.status(400).json({ error: 'Features are required' });
    }
    
    const result = await parkinsonsAPI.predictEnsemble(features);
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Comprehensive assessment endpoint
app.post('/api/parkinsons/comprehensive', async (req, res) => {
  try {
    const testData = req.body;
    
    if (!testData || Object.keys(testData).length === 0) {
      return res.status(400).json({ error: 'Test data is required' });
    }
    
    const result = await parkinsonsAPI.predictComprehensive(testData);
    
    // Example: Save comprehensive results to database
    // await saveComprehensiveResults(req.user.id, result);
    
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Get model features endpoint
app.get('/api/parkinsons/features/:modelName', async (req, res) => {
  try {
    const { modelName } = req.params;
    const features = await parkinsonsAPI.getModelFeatures(modelName);
    res.json(features);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Patient assessment history (example endpoint - implement your database logic)
app.get('/api/parkinsons/history/:patientId', async (req, res) => {
  try {
    const { patientId } = req.params;
    
    // Implement your database query here
    // const history = await getPatientHistory(patientId);
    
    res.json({
      message: 'Implement your database logic here',
      patientId
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({ error: 'Something went wrong!' });
});

// Start server
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
  console.log(`Make sure the Python ML API is running on http://localhost:5000`);
});

module.exports = app;