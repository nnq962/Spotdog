import time
from datetime import datetime

# Ghi lại thời gian bắt đầu
start_time = time.time()

# Chạy vòng lặp 100 lần
for i in range(100):
    pass

# Ghi lại thời gian kết thúc
end_time = time.time()

# Chuyển đổi thời gian bắt đầu và kết thúc thành định dạng ngày tháng năm
start_time_str = datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
end_time_str = datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')

# Tính thời gian chạy bằng cách trừ thời gian kết thúc cho thời gian bắt đầu
run_time = end_time - start_time

print("Thời gian bắt đầu:", start_time_str)
print("Thời gian kết thúc:", end_time_str)
print("Thời gian chạy:", run_time, "giây")
