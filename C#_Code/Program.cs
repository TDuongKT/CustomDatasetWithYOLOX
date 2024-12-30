using Telegram.Bot;

class Program
{
    static string BOT_TOKEN = "7620865045:AAF8odh0Ww6eBgFOyGbwX5Pid_6J8Kh1akQ"; // Thay bằng token của bot Telegram
    static string CHAT_ID = "5997764377";     // Thay bằng chat ID của bạn
    static DateTime lastNotificationTime = DateTime.MinValue; // Thời gian gửi thông báo gần nhất
    static DateTime lastImageSendTime = DateTime.MinValue;     // Thời gian gửi ảnh gần nhất
    static int notificationIntervalInSeconds = 1; // Thời gian giữa các thông báo
    static int imageSendIntervalInSeconds = 5;    // Thời gian giữa các lần gửi ảnh

    // Hàm gửi thông báo
    static async void SendTelegramMessage(string message)
    {
        try
        {
            var botClient = new TelegramBotClient(BOT_TOKEN);
            await botClient.SendTextMessageAsync(CHAT_ID, message);
            Console.WriteLine($"Message sent: {message}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error sending message: {ex.Message}");
        }
    }

    // Hàm gửi file ảnh
    static async void SendTelegramImage(string filePath)
    {
        try
        {
            // Kiểm tra xem file có đang bị khóa không
            if (!IsFileReady(filePath))
            {
                Console.WriteLine($"File is not ready: {filePath}");
                return;
            }

            var botClient = new TelegramBotClient(BOT_TOKEN);
            using (var stream = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                await botClient.SendPhotoAsync(
                    chatId: CHAT_ID,
                    photo: stream,
                    caption: "Phát hiện đối tượng - Hình ảnh!"
                );
                Console.WriteLine($"Image sent: {filePath}");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error sending image: {ex.Message}");
        }
    }

    // Hàm kiểm tra file có sẵn để đọc hay không
    static bool IsFileReady(string filePath)
    {
        try
        {
            using (var stream = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.None))
            {
                return true;
            }
        }
        catch (IOException)
        {
            return false;
        }
    }

    static void Main(string[] args)
    {
        string filePath = @"D:\Honda_PlusVN\Python\Python_Project\YOLOX\Thelf_Detection\image.jpg"; // Đường dẫn file image.jpg
        if (!File.Exists(filePath))
        {
            Console.WriteLine("File does not exist: " + filePath);
            return;
        }

        // Tạo FileSystemWatcher để theo dõi file
        FileSystemWatcher watcher = new FileSystemWatcher
        {
            Path = Path.GetDirectoryName(filePath),
            Filter = Path.GetFileName(filePath),
            NotifyFilter = NotifyFilters.LastWrite // Theo dõi thời gian thay đổi
        };

        watcher.Changed += (sender, e) =>
        {
            DateTime now = DateTime.Now;

            // Gửi thông báo mỗi 1 giây
            if ((now - lastNotificationTime).TotalSeconds >= notificationIntervalInSeconds)
            {
                lastNotificationTime = now;
                SendTelegramMessage("Phát hiện đối tượng!");
            }

            // Gửi file ảnh mỗi 5 giây
            if ((now - lastImageSendTime).TotalSeconds >= imageSendIntervalInSeconds)
            {
                lastImageSendTime = now;
                SendTelegramImage(filePath);
            }
        };

        watcher.EnableRaisingEvents = true; // Bắt đầu theo dõi

        Console.WriteLine("Đang theo dõi file... Nhấn Enter để thoát.");
        Console.ReadLine();
    }
}

