<h2 id="data-preparation-guide">🎯 Hướng Dẫn Chi Tiết Chuẩn Bị Dữ Liệu cho YOLOv10 (WSI/Tế bào học)</h2>

<p>Việc chuẩn bị dữ liệu đúng định dạng là bước quan trọng nhất để huấn luyện mô hình Object Detection (Phát hiện vật thể).</p>

<hr>

<h3 id="step-1-roi-and-image-preparation">1. Chuẩn bị Ảnh ROI và Quy tắc Đặt tên</h3>

<h4>1.1. Chuẩn bị Ảnh (ROI - Region of Interest)</h4>
<ul>
    <li><strong>Kích thước ảnh:</strong> YOLOv4 hoạt động tốt nhất với các ảnh có kích thước được chia hết cho 32. Kích thước <code>608x608px</code> bạn chọn là lý tưởng và là kích thước chuẩn thường dùng cho YOLO.</li>
    <li><strong>Định dạng ảnh:</strong> Nên sử dụng định dạng <code>.jpg</code> hoặc <code>.png</code>. (<code>.jpg</code> thường nhẹ hơn và được ưu tiên).</li>
</ul>

<h4>1.2. Quy tắc Đặt tên File Ảnh và Nhãn (Rất Quan trọng)</h4>
<p>Quy tắc cơ bản nhất là: <strong>Mỗi tệp ảnh phải có một tệp nhãn tương ứng, có cùng tên gốc, nhưng khác phần mở rộng (extension).</strong></p>
<table>
    <thead>
        <tr>
            <th>Thành phần</th>
            <th>Tên Gốc (Ví dụ)</th>
            <th>Phần Mở rộng</th>
            <th>Mục đích</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Tệp Ảnh</td>
            <td><code>lam_123_vung_A</code></td>
            <td><code>.jpg</code> (hoặc <code>.png</code>)</td>
            <td>Hình ảnh cần phát hiện vật thể.</td>
        </tr>
        <tr>
            <td>Tệp Nhãn</td>
            <td><code>lam_123_vung_A</code></td>
            <td><code>.txt</code></td>
            <td>Chứa thông tin vị trí và lớp của vật thể trong ảnh trên.</td>
        </tr>
    </tbody>
</table>
<p class="note"><strong>Ghi chú:</strong> Không cần đặt tên biến hay định nghĩa phức tạp cho các tệp ảnh/nhãn; chỉ cần đảm bảo sự đồng bộ về tên gốc.</p>

<hr>

<h3 id="step-2-labeling-file-format">2. Định dạng File Nhãn (<code>.txt</code>)</h3>

<p>Mỗi tệp <code>.txt</code> tương ứng với một ảnh sẽ chứa thông tin của **tất cả** các vật thể trong ảnh đó. Mỗi vật thể (hộp giới hạn) là một dòng riêng biệt.</p>

<h4>2.1. Định dạng Chuẩn YOLO</h4>
<p>Mỗi dòng tuân theo cú pháp sau, sử dụng **dấu cách** (space) để phân tách:</p>
<pre><code>[class_id] [x_center] [y_center] [width] [height]</code></pre>

<h4>2.2. Chi tiết các tham số</h4>
<ul>
    <li>
        <strong><code>class_id</code>:</strong> 
        <ul>
            <li>Là một số nguyên (integer) **bắt đầu từ 0**.</li>
            <li>Đây là chỉ mục của lớp đó trong tệp <code>obj.names</code> (ví dụ: nếu "viem" là dòng đầu tiên trong <code>obj.names</code>, <code>class_id</code> của nó là <code>0</code>).</li>
        </ul>
    </li>
    <li>
        <strong><code>x_center</code>, <code>y_center</code>, <code>width</code>, <code>height</code>:</strong> 
        <ul>
            <li>Là các số thực (float) được **chuẩn hóa (normalized)**, nằm trong khoảng <code>0.0</code> đến <code>1.0</code>.</li>
            <li><code>x_center</code> và <code>width</code> được chuẩn hóa theo chiều rộng của ảnh.</li>
            <li><code>y_center</code> và <code>height</code> được chuẩn hóa theo chiều cao của ảnh.</li>
            <li><strong>Mục đích chuẩn hóa:</strong> Giúp mô hình hoạt động độc lập với kích thước ảnh gốc.</li>
        </ul>
    </li>
</ul>

<p><strong>Ví dụ File <code>lam_123_vung_A.txt</code>:</strong> (Giả sử Lớp 'viem' là ID 0, Lớp 'bat_thuong' là ID 1)</p>
<pre><code>0 0.500000 0.500000 0.100000 0.200000  &lt;-- Vật thể 1: Lớp 'viem'
1 0.250000 0.750000 0.050000 0.080000  &lt;-- Vật thể 2: Lớp 'bat_thuong'
...</code></pre>

<hr>

<h3 id="step-3-configuration-files">3. Chuẩn bị các Tệp Cấu hình (Configuration Files)</h3>

<h4>3.1. <code>obj.names</code></h4>
<ul>
    <li><strong>Mục đích:</strong> Định nghĩa tên của các lớp mà mô hình cần phát hiện.</li>
    <li><strong>Định dạng:</strong> Mỗi tên lớp trên một dòng riêng biệt.</li>
    <li><strong>Thứ tự:</strong> Thứ tự này phải **trùng khớp** với <code>class_id</code> trong tệp nhãn <code>.txt</code>.</li>
</ul>
<p><strong>Nội dung <code>obj.names</code> (Ví dụ):</strong></p>
<pre><code>te_bao_bat_thuong
ton_thuong_tuyen
viem
...</code></pre>
<p class="note"><strong>Lưu ý:</strong> Trong ví dụ này, <code>te_bao_bat_thuong</code> có <code>class_id = 0</code>, <code>ton_thuong_tuyen</code> có <code>class_id = 1</code>, v.v.</p>

<h4>3.2. <code>obj.data</code></h4>
<ul>
    <li><strong>Mục đích:</strong> Tệp "siêu dữ liệu" chỉ cho Darknet biết vị trí của các tệp quan trọng khác và các thông số cần thiết.</li>
</ul>
<p><strong>Nội dung <code>obj.data</code> (Ví dụ):</strong></p>
<pre><code>classes= 3                   # Số lượng lớp (classes) của bạn
train  = data/train.txt      # Đường dẫn đến file chứa danh sách ảnh TRAIN
valid  = data/valid.txt      # Đường dẫn đến file chứa danh sách ảnh VALID (tùy chọn)
names  = data/obj.names      # Đường dẫn đến file obj.names
backup = /mydrive/yolov4_backup/  # Vị trí Colab lưu trọng số đã huấn luyện (checkpoint)
</code></pre>
<p class="note"><strong>Lưu ý:</strong> Bạn cần tạo thêm 2 tệp <code>train.txt</code> và <code>valid.txt</code>. Mỗi tệp này chứa đường dẫn (tương đối hoặc tuyệt đối) của mỗi ảnh dùng cho huấn luyện và kiểm thử.</p>

<h4>3.3. <code>yolov4-custom.cfg</code></h4>
<ul>
    <li><strong>Mục đích:</strong> Định nghĩa kiến trúc của mạng nơ-ron.</li>
    <li><strong>Cách làm:</strong> Bạn nên sao chép tệp <code>yolov4-custom.cfg</code> có sẵn trong Darknet và CHỈ chỉnh sửa các dòng sau:
        <ul>
            <li>Ở phần <code>[net]</code> (đầu file):
                <pre><code>batch=64
subdivisions=16
height=608
width=608</code></pre>
            </li>
            <li>Ở **3 khối <code>[yolo]</code> cuối cùng** (cuối file):
                <ul>
                    <li>**<code>classes=</code>**: Đặt bằng số lượng lớp của bạn (ví dụ: <code>classes=3</code>).</li>
                </ul>
            </li>
            <li>Ở **3 khối <code>[convolutional]</code> ngay phía trước <code>[yolo]</code>**:
                <ul>
                    <li>**<code>filters=</code>**: Đặt bằng công thức: $3 \times (5 + \text{classes})$ (ví dụ: $3 \times (5 + 3) = 24$).</li>
                </ul>
            </li>
        </ul>
    </li>
</ul>

<hr>

<h3 id="step-4-uploading-to-colab">4. Tải Dữ liệu lên Colab (Sử dụng Google Drive)</h3>

<ul>
    <li><strong>Tổ chức:</strong> Đặt tất cả ảnh, nhãn, và các tệp cấu hình (<code>.names</code>, <code>.data</code>, <code>.cfg</code>, <code>train.txt</code>, <code>valid.txt</code>) vào một thư mục gốc.</li>
    <li><strong>Nén File:</strong> Nén thư mục gốc đó thành một tệp <code>.zip</code> (ví dụ: <code>my_yolo_data.zip</code>).</li>
    <li><strong>Tải lên Drive:</strong> Tải tệp <code>.zip</code> này lên Google Drive cá nhân của bạn.</li>
    <li><strong>Mount Drive trong Colab:</strong> Dùng lệnh sau trong Colab để truy cập Drive:
        <pre><code>from google.colab import drive
drive.mount('/mydrive')</code></pre>
    </li>
    <li><strong>Giải nén Dữ liệu:</strong> Sau đó, di chuyển hoặc giải nén dữ liệu vào thư mục <code>/content/darknet/data/</code> (hoặc bất kỳ vị trí nào mà bạn chỉ định trong <code>obj.data</code>).
        <pre><code>!unzip /mydrive/path/to/my_yolo_data.zip -d /content/darknet/data/</code></pre>
    </li>
</ul>

<p>Việc sử dụng Google Drive giúp bạn lưu trữ dữ liệu lớn (ảnh WSI) ổn định mà không bị mất khi phiên Colab bị ngắt.</p># YOLOv4
