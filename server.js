const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const cors = require('cors');
const axios = require('axios');
const FormData = require('form-data');
const { spawn } = require('child_process');

const app = express();
app.use(cors());
app.use(express.static(path.join(__dirname, 'public')));  // Serving static files (e.g., JS, CSS)
app.use(express.static(__dirname));  // Serve files from the root directory, where .glb files are located
app.use('/csv-files', express.static(path.join(__dirname, 'models/results/csv')));

// Create necessary directories for uploads
const createRequiredDirs = async () => {
    const dirs = ['uploads', 'models'];
    for (const dir of dirs) {
        const dirPath = path.join(__dirname, dir);
        try {
            await fs.promises.mkdir(dirPath, { recursive: true });
        } catch (error) {
            console.error(`Error creating directory ${dir}:`, error);
        }
    }
};
createRequiredDirs();

// Multer configuration for file uploads
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, path.join(__dirname, 'uploads'));
    },
    filename: (req, file, cb) => {
        const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1e9);
        cb(null, file.fieldname + '-' + uniqueSuffix + path.extname(file.originalname));
    }
});

const upload = multer({
    storage,
    limits: { fileSize: 20 * 1024 * 1024 } // Set file size limit to 20MB
});

// Helper function to interact with Python API
async function getErgonomicMeasurements(beforeImagePath, afterImagePath) {
    const formData = new FormData();
    formData.append('sitting_image', fs.createReadStream(beforeImagePath));
    formData.append('standing_image', fs.createReadStream(afterImagePath));

    try {
        console.log("Sending images to Python API...");
        const response = await axios.post('http://127.0.0.1:5001/analyze/ergonomics', formData, {
            headers: { ...formData.getHeaders() },
            maxBodyLength: Infinity,
            maxContentLength: Infinity
        });
        console.log("Received response from Python API");
        return response.data;
    } catch (error) {
        console.error('Error calling Python API:', error.message);
        throw error;
    }
}

async function modify3DModel(csvFilePath) {
    const blenderExecutable = '/Applications/Blender.app/Contents/MacOS/Blender';
    const pythonScriptPath = path.join(__dirname, 'scripts', 'ModifyModel.py');

    const glbFileName = 'Scaled3dModel.glb';  // Name for the output .glb file
    const glbFilePath = path.join(__dirname, glbFileName);

    // Spawn Blender process to run the Python script
    return new Promise((resolve, reject) => {
        const blenderProcess = spawn(blenderExecutable, [
            '--background',  // Run Blender in background (no GUI)
            '--python', pythonScriptPath,  // Path to the Python script
            '--', csvFilePath  // Pass the CSV file path as an argument to the Python script
        ]);

        // Capture Blender's output
        blenderProcess.stdout.on('data', (data) => {
            console.log(`stdout: ${data}`);
        });

        blenderProcess.stderr.on('data', (data) => {
            console.error(`stderr: ${data}`);
        });

        blenderProcess.on('close', (code) => {
            if (code !== 0) {
                console.error(`Blender process exited with code ${code}`);
                reject(new Error(`Blender process failed with code ${code}`));  // Reject on failure
            } else {
                console.log('3D model modified successfully');
                resolve(glbFilePath);  // Resolve the promise with the correct file path
            }
        });
    });
}

// Upload endpoint for processing images
app.post('/upload-images', upload.fields([
    { name: 'beforeImage', maxCount: 1 },
    { name: 'afterImage', maxCount: 1 }
]), async (req, res) => {
    const uploadedFiles = [];
    try {
        if (!req.files || !req.files.beforeImage || !req.files.afterImage) {
            return res.status(400).json({
                success: false,
                error: 'Both before and after images are required.'
            });
        }

        const beforeImage = req.files.beforeImage[0].path;
        const afterImage = req.files.afterImage[0].path;
        uploadedFiles.push(beforeImage, afterImage);

        // Call Python API to process the images
        const apiResponse = await getErgonomicMeasurements(beforeImage, afterImage);

        // Python service is responsible for saving results
        console.log(`Results saved successfully at:\n- JSON: ${apiResponse.json_file}\n- CSV: ${apiResponse.csv_file}`);

        const csvFilePath = apiResponse.csv_file; // Path to the saved CSV
        const glbFilePath = await modify3DModel(csvFilePath);

        console.log('Generated glbFilePath:', glbFilePath); // Log to verify it's being returned

        if (!glbFilePath) {
            throw new Error('GLB file path is undefined or invalid');
        }

        // Send the correct URL for the GLB file
        res.json({
            success: true,
            message: 'Processing complete. Results saved by the Python service.',
            modelUrl: `http://localhost:3000/${path.basename(glbFilePath)}`, // Path to GLB file
            results: {
                json_file: apiResponse.json_file,
                glb_file: glbFilePath
            }
        });

        // Clean up uploaded files
        await Promise.all(uploadedFiles.map(file => fs.promises.unlink(file)));
    } catch (error) {
        console.error('Error processing upload:', error);

        // Clean up uploaded files in case of error
        await Promise.all(uploadedFiles.map(async (file) => {
            try {
                await fs.promises.unlink(file);
            } catch (e) {
                console.error(`Error deleting file ${file}:`, e);
            }
        }));

        res.status(500).json({
            success: false,
            error: 'Error processing the images',
            details: error.message
        });
    }
});

// Health check endpoint
app.get('/health', async (req, res) => {
    try {
        const response = await axios.get('http://127.0.0.1:5001/health');
        res.json({
            success: true,
            message: 'Server is healthy and Python API connection is working',
            pythonApiStatus: response.data.status
        });
    } catch (error) {
        console.error('Health check failed:', error.message);
        res.status(500).json({
            success: false,
            error: 'Health check failed. Python API might not be running.',
            details: error.message
        });
    }
});

// Error handling middleware
app.use((err, req, res, next) => {
    console.error('Unhandled error:', err.message);
    res.status(500).json({
        success: false,
        error: 'Internal server error',
        details: err.message
    });
});

// Start server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});
