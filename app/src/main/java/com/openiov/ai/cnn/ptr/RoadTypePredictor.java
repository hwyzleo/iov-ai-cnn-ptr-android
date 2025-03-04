package com.openiov.ai.cnn.ptr;

import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.os.Build;
import android.util.Log;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

import java.nio.FloatBuffer;
import java.util.Collections;

public class RoadTypePredictor {
    private static final int INPUT_SIZE = 360;
    private static final String[] ROAD_TYPES = {
            "road_type_asphalt",
            "road_type_concrete",
            "road_type_gravel",
            "road_type_dirt",
            "road_type_unknown"
    };

    private final OrtEnvironment env;
    private final OrtSession session;

    public RoadTypePredictor(String modelPath) throws Exception {
        env = OrtEnvironment.getEnvironment();
        
        // 检查设备芯片型号并配置执行提供程序
        String deviceSoc = Build.HARDWARE;
        Log.d("RoadTypePredictor", "设备芯片型号: " + deviceSoc);
        
        OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();
        
        // 检查是否支持AI加速
        boolean isAiAccelerationSupported = deviceSoc != null && (
            deviceSoc.contains("lahaina") || // Snapdragon 888
            deviceSoc.contains("taro") ||    // Snapdragon 8 Gen 1
            deviceSoc.contains("kalama") ||  // Snapdragon 8 Gen 2
            deviceSoc.contains("sm8150") || // Snapdragon 855
            deviceSoc.contains("sm8250") || // Snapdragon 865
            deviceSoc.contains("sm8350") || // Snapdragon 888
            deviceSoc.contains("sm8450") || // Snapdragon 8 Gen 1
            deviceSoc.contains("sm8550")    // Snapdragon 8 Gen 2
        );
        
        if (isAiAccelerationSupported) {
            Log.d("RoadTypePredictor", "设备支持AI加速，启用NNAPI执行提供程序");
            sessionOptions.addConfigEntry("session.use_nnapi", "1");
        } else {
            Log.d("RoadTypePredictor", "设备不支持AI加速，使用默认CPU执行提供程序");
        }
        
        session = env.createSession(modelPath, sessionOptions);
    }

    public String predict(Bitmap image) throws Exception {
        Log.d("RoadTypePredictor", "开始处理输入图像，原始尺寸: " + image.getWidth() + "x" + image.getHeight());
        Bitmap resizedImage = resizeImage(image);
        Log.d("RoadTypePredictor", "图像缩放完成，新尺寸: " + resizedImage.getWidth() + "x" + resizedImage.getHeight());
        
        float[] inputArray = preprocessImage(resizedImage);
        Log.d("RoadTypePredictor", "图像预处理完成，输入数组大小: " + inputArray.length);
        FloatBuffer inputBuffer = FloatBuffer.wrap(inputArray);

        long[] shape = {1, 3, INPUT_SIZE, INPUT_SIZE};
        Log.d("RoadTypePredictor", "创建输入张量，形状: [1, 3, " + INPUT_SIZE + ", " + INPUT_SIZE + "]");
        OnnxTensor inputTensor = OnnxTensor.createTensor(env, inputBuffer, shape);
        Log.d("RoadTypePredictor", "开始执行模型推理");

        float[] output = ((OnnxTensor) session.run(
                Collections.singletonMap("input", inputTensor)
        ).get(0)).getFloatBuffer().array();
        Log.d("RoadTypePredictor", "获取模型输出，分数数组大小: " + output.length);

        int maxIndex = 0;
        float maxValue = output[0];
        for (int i = 1; i < output.length; i++) {
            if (output[i] > maxValue) {
                maxValue = output[i];
                maxIndex = i;
            }
        }

        return ROAD_TYPES[maxIndex];
    }

    private Bitmap resizeImage(Bitmap image) {
        float scale = (float) INPUT_SIZE / Math.min(image.getWidth(), image.getHeight());
        Matrix matrix = new Matrix();
        matrix.postScale(scale, scale);

        return Bitmap.createBitmap(
                image,
                0, 0,
                image.getWidth(), image.getHeight(),
                matrix, true);
    }

    private float[] preprocessImage(Bitmap image) {
        int[] pixels = new int[INPUT_SIZE * INPUT_SIZE];
        image.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);

        float[] result = new float[3 * INPUT_SIZE * INPUT_SIZE];
        int idx = 0;
        for (int c = 0; c < 3; c++) {
            for (int i = 0; i < INPUT_SIZE; i++) {
                for (int j = 0; j < INPUT_SIZE; j++) {
                    int pixel = pixels[i * INPUT_SIZE + j];
                    float value = ((pixel >> ((2 - c) * 8)) & 0xFF) / 255.0f;
                    result[idx++] = (value - 0.5f) * 2.0f;
                }
            }
        }
        return result;
    }

    public void close() {
        try {
            session.close();
            env.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}