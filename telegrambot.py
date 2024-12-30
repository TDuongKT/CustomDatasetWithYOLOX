from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

BOT_TOKEN = (
    "7620865045:AAF8odh0Ww6eBgFOyGbwX5Pid_6J8Kh1akQ"  # Thay bằng token của bot Telegram
)


# Hàm xử lý lệnh /start để lấy chat_id
async def get_chat_id(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    await update.message.reply_text(f"Chat ID của bạn là: {chat_id}")


# Hàm chính để khởi tạo bot
def main():
    # Tạo ứng dụng bot
    application = Application.builder().token(BOT_TOKEN).build()

    # Thêm handler cho lệnh /start
    application.add_handler(CommandHandler("start", get_chat_id))

    # Bắt đầu polling
    application.run_polling()


if __name__ == "__main__":
    main()
