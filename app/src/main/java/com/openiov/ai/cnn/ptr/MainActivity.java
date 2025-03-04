package com.openiov.ai.cnn.ptr;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.SurfaceTexture;
import android.media.MediaMetadataRetriever;
import android.media.MediaPlayer;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.view.Surface;
import android.view.TextureView;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.VideoView;
import android.view.View;
import android.view.ViewGroup;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.app.ActivityManager;
import android.content.Context;
import android.os.Debug;
import android.os.Process;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.File;
import java.io.FileDescriptor;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

public class MainActivity extends AppCompatActivity {
    private static final int PERMISSION_REQUEST_CODE = 1;
    private static final String MODEL_FILE = "model.onnx";
    private static final String TEST_VIDEO_FILE = "test_video.mp4";

    private VideoView previewView;
    private TextView resultText;
    private Button controlButton;
    private MediaPlayer mediaPlayer;
    private boolean isInferencing = false;
    private ScheduledExecutorService inferenceExecutor;
    private RoadTypePredictor predictor;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        previewView = findViewById(R.id.preview_view);
        resultText = findViewById(R.id.result_text);
        controlButton = findViewById(R.id.control_button);

        previewView.setOnPreparedListener(mp -> {
            mp.setLooping(true);
            if (isInferencing) {
                mp.start();
            }
        });
        controlButton.setOnClickListener(v -> toggleInference());

        checkAndRequestPermissions();
        copyAssetsToFiles();
        initPredictor();
        initVideoPlayer();
    }

    private void checkAndRequestPermissions() {
        String[] permissions = {
                Manifest.permission.READ_EXTERNAL_STORAGE,
                Manifest.permission.WRITE_EXTERNAL_STORAGE,
                Manifest.permission.CAMERA
        };

        for (String permission : permissions) {
            if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this, permissions, PERMISSION_REQUEST_CODE);
                break;
            }
        }
    }

    private void copyAssetsToFiles() {
        copyAssetToFile(MODEL_FILE);
        copyAssetToFile(TEST_VIDEO_FILE);
    }

    private void copyAssetToFile(String filename) {
        File file = new File(getFilesDir(), filename);
        Log.d("MainActivity", "正在复制文件: " + filename + " 到 " + file.getAbsolutePath());

        try (InputStream is = getAssets().open(filename);
             FileOutputStream os = new FileOutputStream(file)) {
            byte[] buffer = new byte[4096];
            int read;
            long total = 0;
            while ((read = is.read(buffer)) != -1) {
                os.write(buffer, 0, read);
                total += read;
            }
            Log.d("MainActivity", "文件复制完成，大小: " + total + " 字节");
        } catch (IOException e) {
            Log.e("MainActivity", "复制文件失败: " + e.getMessage());
            e.printStackTrace();
            Toast.makeText(this, "Failed to copy " + filename, Toast.LENGTH_SHORT).show();
        }
    }

    private void initPredictor() {
        try {
            File modelFile = new File(getFilesDir(), MODEL_FILE);
            Log.d("MainActivity", "开始初始化预测器，模型文件路径: " + modelFile.getAbsolutePath());
            if (!modelFile.exists()) {
                Log.e("MainActivity", "模型文件不存在");
                Toast.makeText(this, "Model file not found", Toast.LENGTH_SHORT).show();
                return;
            }
            predictor = new RoadTypePredictor(modelFile.getAbsolutePath());
            Log.d("MainActivity", "预测器初始化成功");
        } catch (Exception e) {
            Log.e("MainActivity", "初始化预测器失败: " + e.getMessage());
            e.printStackTrace();
            Toast.makeText(this, "Failed to initialize predictor", Toast.LENGTH_SHORT).show();
        }
    }

    private void toggleInference() {
        if (isInferencing) {
            stopInference();
        } else {
            startInference();
        }
    }

    private void startInference() {
        Log.d("MainActivity", "开始推理流程");
        if (predictor == null) {
            Log.e("MainActivity", "predictor为空，无法开始推理");
            return;
        }

        runOnUiThread(() -> {
            isInferencing = true;
            controlButton.setText(R.string.stop_inference);
            resultText.setText(R.string.processing);
            if (!previewView.isPlaying()) {
                Log.d("MainActivity", "开始播放视频");
                previewView.start();
            }
        });

        inferenceExecutor = Executors.newSingleThreadScheduledExecutor();
        inferenceExecutor.scheduleAtFixedRate(() -> {
            if (!isInferencing) return;

            try {
                int currentPosition = previewView.getCurrentPosition();
                Log.d("MainActivity", "当前视频位置: " + currentPosition + "ms");
                MediaMetadataRetriever retriever = new MediaMetadataRetriever();
                String videoPath = new File(getFilesDir(), TEST_VIDEO_FILE).getAbsolutePath();
                Log.d("MainActivity", "视频文件路径: " + videoPath);
                retriever.setDataSource(videoPath);
                Bitmap frame = retriever.getFrameAtTime(currentPosition * 1000, MediaMetadataRetriever.OPTION_CLOSEST);
                Log.d("MainActivity", "成功获取视频帧，尺寸: " + frame.getWidth() + "x" + frame.getHeight());
                retriever.release();

                if (frame != null) {
                    Log.d("MainActivity", "开始对视频帧进行推理");
                    long inferenceStartTime = System.nanoTime();
                    String result = predictor.predict(frame);
                    long inferenceEndTime = System.nanoTime();
                    double inferenceTime = (inferenceEndTime - inferenceStartTime) / 1_000_000.0; // 转换为毫秒
                    Log.d("MainActivity", "推理结果: " + result + ", 耗时: " + inferenceTime + "ms");
                    String performanceInfo = getPerformanceInfo(inferenceTime);
                    Log.d("MainActivity", "性能数据: " + performanceInfo);
                    runOnUiThread(() -> resultText.setText(getString(R.string.inference_result, result) + "\n" + performanceInfo));
                    frame.recycle();
                } else {
                    Log.e("MainActivity", "获取视频帧失败");
                }
            } catch (Exception e) {
                Log.e("MainActivity", "推理过程发生异常: " + e.getMessage());
                e.printStackTrace();
                runOnUiThread(() -> resultText.setText(R.string.road_type_unknown));
            }
        }, 0, 500, TimeUnit.MILLISECONDS);
    }

    private void stopInference() {
        isInferencing = false;
        controlButton.setText(R.string.start_inference);
        if (inferenceExecutor != null) {
            inferenceExecutor.shutdown();
            inferenceExecutor = null;
        }
        resultText.setText(R.string.road_type_unknown);
        previewView.pause();
    }

    private void initVideoPlayer() {
        try {
            File videoFile = new File(getFilesDir(), TEST_VIDEO_FILE);
            Log.d("MainActivity", "开始初始化VideoView，视频文件路径: " + videoFile.getAbsolutePath());
            if (!videoFile.exists()) {
                Log.e("MainActivity", "视频文件不存在");
                Toast.makeText(this, "Video file not found", Toast.LENGTH_SHORT).show();
                return;
            }

            Uri videoUri = Uri.fromFile(videoFile);
            previewView.setVideoURI(videoUri);
            
            // 设置VideoView的宽高比例
            MediaMetadataRetriever retriever = new MediaMetadataRetriever();
            retriever.setDataSource(videoFile.getAbsolutePath());
            int videoWidth = Integer.parseInt(retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH));
            int videoHeight = Integer.parseInt(retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT));
            retriever.release();
            
            previewView.post(() -> {
                int containerWidth = ((View)previewView.getParent()).getWidth();
                int containerHeight = ((View)previewView.getParent()).getHeight();
                
                float videoRatio = (float)videoWidth / videoHeight;
                float containerRatio = (float)containerWidth / containerHeight;
                
                ViewGroup.LayoutParams params = previewView.getLayoutParams();
                if (videoRatio > containerRatio) {
                    // 视频比较宽，以容器宽度为基准
                    params.width = containerWidth;
                    params.height = (int)(containerWidth / videoRatio);
                } else {
                    // 视频比较高，以容器高度为基准
                    params.height = containerHeight;
                    params.width = (int)(containerHeight * videoRatio);
                }
                previewView.setLayoutParams(params);
            });
            
            previewView.setOnErrorListener((mp, what, extra) -> {
                Log.e("MainActivity", "VideoView错误: what=" + what + ", extra=" + extra);
                Toast.makeText(MainActivity.this, "Video playback error", Toast.LENGTH_SHORT).show();
                return true;
            });
            Log.d("MainActivity", "VideoView初始化成功");
        } catch (Exception e) {
            Log.e("MainActivity", "初始化VideoView失败: " + e.getMessage());
            e.printStackTrace();
            Toast.makeText(this, "Failed to play video", Toast.LENGTH_SHORT).show();
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (previewView != null) {
            previewView.pause();
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (previewView != null && isInferencing) {
            previewView.resume();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        stopInference();
        if (predictor != null) {
            predictor.close();
            predictor = null;
        }
    }
    private String getPerformanceInfo(double inferenceTime) {
        ActivityManager activityManager = (ActivityManager) getSystemService(Context.ACTIVITY_SERVICE);
        ActivityManager.MemoryInfo memoryInfo = new ActivityManager.MemoryInfo();
        activityManager.getMemoryInfo(memoryInfo);

        Debug.MemoryInfo debugMemoryInfo = new Debug.MemoryInfo();
        Debug.getMemoryInfo(debugMemoryInfo);

        int pid = Process.myPid();
        int[] pids = {pid};
        Debug.MemoryInfo[] processMemoryInfo = activityManager.getProcessMemoryInfo(pids);
        double totalPss = processMemoryInfo[0].getTotalPss() / 1024.0;

        double availableMem = memoryInfo.availMem / (1024.0 * 1024.0);
        double totalMem = memoryInfo.totalMem / (1024.0 * 1024.0);
        double memoryUsage = ((totalMem - availableMem) / totalMem) * 100;

        double cpuUsage = 0.0;
        try {
            long startTime = System.nanoTime();
            long startCpuTime = Debug.threadCpuTimeNanos();
            Thread.sleep(100); // 采样间隔100ms
            long endTime = System.nanoTime();
            long endCpuTime = Debug.threadCpuTimeNanos();

            long timeDiff = endTime - startTime;
            long cpuTimeDiff = endCpuTime - startCpuTime;

            if (timeDiff > 0) {
                cpuUsage = ((double) cpuTimeDiff / timeDiff) * 100;
            }
        } catch (Exception e) {
            Log.e("MainActivity", "获取CPU使用率失败: " + e.getMessage());
        }

        return String.format("推理耗时: %.1f ms\n内存使用: %.1f MB (PSS)\n系统内存: %.1f%%使用\nCPU使用率: %.1f%%", inferenceTime, totalPss, memoryUsage, cpuUsage);
    }
}