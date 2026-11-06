# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`= 468 ; `class_name`= "cab" ; `rank`=1 ; `score`= 0.510915 ; `taxonomy_name`= "ImageNet-1K"
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy? Record này nói về cả bức ảnh. Vì sử dụng mô hình YOLO ở chế độ classification, đưa toàn bộ ảnh vào mô hình dự đoán và lấy xác suất dự đoán các lớp của toàn bộ ảnh.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint? Danh sách tên lớp đi kèm taxonomy/checkpoint. Vì trong đoạn code có names = result.names, lấy danh sách tên các lớp từ YOLO. 
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp? Vì mỗi trường có vai trò khác nhau. class_id để máy xử lí nhanh, class_name để người đọc hiểu, taxonomy_name xác định lớp đó thuộc hệ thống phân loại nào.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp? Guideline cần nói rõ:
- Vật/chủ đề chính: Chọn đối tượng chiếm vị trí trung tâm hoặc nổi bật nhất trong ảnh
- Mức độ chiếm diện tích: Chọn vật thể chiếm diện tích lớn nhất trong ảnh
- Nếu không xác định được: Quy định nhãn "Other"
6. Vì sao không được lấy `score` của model làm ground truth? Vì score chỉ thể hiện mức độ tự tin của mô hình đối với dự đoán, không chứng minh dự đoán đó là đúng.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`= "bowl" ; `score`= 0.719063 ; `bbox_xyxy`=[32.65, 342.12, 100.16, 384.93] ; `bbox_width`= 67.51 ; `bbox_height`= 42.81
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel? Góc trên-trái ở pixel [32.65, 342.12], góc dưới-phải ở pixel [100.16, 384.93], rộng 67.51 pixel, cao 42.81 pixel.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction? Threshold=0.35: 11 vật thể; threshold=0.60: 6 vật thể
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì? 
- Kiểm tra thêm các box có confidence thấp.
- Loại bỏ box sai, box trùng lặp.
- Kiểm tra lại class và chỉnh sửa bounding box.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
Nếu vật thể bị che khuất hoặc cắt khỏi mép ảnh, chỉ vẽ box bao quanh phần vật thể thực sự nhìn thấy, không tự suy đoán hoặc vẽ thêm phần bị khuất.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
- Vật bị cắt mép: person 0.61
- Em sẽ đưa vào guidline. Vì vật bị cắt mép, chỉ vẽ box quanh phần nhìn thấy được, trường hợp này khá phổ biến, nên được đưa vào guideline.
## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`= "kitchen-011" ; `class_name`= "spoon" ; `score`= 0.359147 ; số điểm polygon=`polygon_point_count`= 37 ; 3–5 điểm đầu của `polygon_xy`= [529.0,66.0], [529.0, 72.0],
 [528.0, 73.0]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
Polygon cho thêm thông tin đường biên thực tế của vật thể bằng một tập hợp các điểm tọa độ nối với nhau.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
- instance_id là id riêng cho từng đối tượng được phát hiện trong một ảnh.
- Em hiểu câu trên: instance_id khác class_id và Tracking_id. class_id là mã loại đối tượng. instance_id mà mã phân biệt từng đối tượng cụ thể trong ảnh. tracking_id là mã theo dõi cùng một đối tượng qua nhiều khung hình video.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
Người gán nhãn phải vẽ polygon bám sát đường viền thực tế của phần vật thể nhìn thấy được, bao gồm các đường cong và góc cạnh, không bao gồm phần nền hoặc vật thể khác.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
- Khu vực bị che: các chiếc bát ở góc trái bên dưới ảnh.
- Em sẽ escalation lên Lab Coach nếu guideline chưa quy định về trường hợp này. Vì cần Lab Coach đưa ra quyết định thống nhất.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh |Một ảnh được gán một class_id, class_name theo taxonomy.|Ảnh kitchen có nhiều đối tượng như người, bát, thìa, bàn. Khó quyết định lớp đại diện nếu guideline yêu cầu chỉ chọn một lớp.|Xác định nội dung chính của ảnh, đối chiếu taxonomy và guideline, gán nhãn đúng quy tắc.|Kiểm tra class có đúng taxonomy, nhất quán với nội dung ảnh và quy tắc chọn lớp không.|
| Phát hiện vật thể |Mỗi vật thể có một nhãn class_id, class_name và một bounding box|Nhiều vật thể bowl ở góc dưới bên trái gần nhau, một số bị che khuất|Vẽ box sát đường viền phần vật thể nhìn thấy, chọn đúng class, không bỏ sót hoặc gộp các vật thể riêng biệt.|Kiểm tra class, vị trí và kích thước box, box trùng lặp, vật thể bị bỏ sót hoặc box bao gồm quá nhiều nền.|
| Instance segmentation |Mỗi instance có instance_id, class_id, class_name và polygon/mask mô tả vùng pixel thuộc về vật thể.|PNG cho thấy các mask của nhiều chiếc bát ở góc dưới bên trái gần nhau, ranh giới giữa bát và mặt bàn khó xác định|Vẽ polygon riêng cho từng vật thể, bám sát biên phần nhìn thấy được, không gộp hai instance cùng class.|Kiểm tra mask có đúng hình dạng, có lấn sang vật thể khác, thiếu vùng, thừa vùng hoặc nhầm instance không.|

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
Chỉ sử dụng ảnh CÔC công khai có sẵn, không tải ảnh cá nhân hay dữ liệu nội bộ. 
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
Em sẽ dừng xử lý, không tự ý gán nhãn và đồng thời báo cáo cho Lab Coach để được hướng dẫn xử lý.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
