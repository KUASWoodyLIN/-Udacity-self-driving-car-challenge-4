# Advanced Lane Finding Project

### 專案路徑布局

```
├── Advanced_Lane_Lines.py
├── project_video.mp4				# Project 影像
├── challenge_video.mp4				# 挑戰 1) 影像
├── harder_challenge_video.mp4		# 挑戰 2) 影像
├── __init__.py	
├── LICENSE
├── image_processing				# 所有pipline中使用到的function都在此資料夾底下
│   ├── camera_cal					# 校正相機使用的影像
│   │   └── .....					
│   ├── calibration.py				# 校正相機function
│   ├── wide_dist_pickle.p			# 存放校正相機後得到cameraMatrix & distCoeffs數值
│   ├── edge_detection.py			# 檢測邊緣function,像sobel,color transforms等等 
│   ├── find_lines.py				# 找出車到線function
│   ├── transform.py				# 轉換測試程式,用來調整測試參數用
│   ├── line_fit_fix.py				# 檢測找車道function是否有正確找出車道
│   ├── preprocessing.py			# 影像遮罩，將不重要地方遮罩掉
│   └── __init__.py	
├── examples						# 執行結果範例
│   └── .....
├── test_images						# 測試影片
│   └── .....
├── output_images					# 測試影片輸出資料夾
│   └── .....
├── output_video					# 測試影像輸出資料夾
│   ├── project_video_long_line.avi		# 使用較長的道路檢測
│   └── project_video_short_line.avi	# 使用較短的道路建測
└── README.md

```

---

### Camera Calibration

程式位置： [calibration.py](./image_processing/calibration.py)

裡面分為兩個function：`found_chessboard()`, `camera_cal()`

`found_chessboard()`: 用來找出棋盤的corners。
`camera_cla()`: 可以將影像undistorted。

下圖為校正前和校正後的比較圖，左圖為原始圖，又圖為校正後結果

![校正前後比較](/home/woodylin/tensorflow3/Udacity_self_driving_car/Udacity_self_driving_car_challenge_4/output_images/undistort_compare.png)

---

### Edge Detection

程式位置： [edge_detection.py](./image_processing/edge_detection.py)

