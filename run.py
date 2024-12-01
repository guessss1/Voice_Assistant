import gradio as gr
import whisper
import openai
from dotenv import load_dotenv
import os
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Загрузка API ключа из .env файла
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

starting_prompt = """Вы — ассистент.
Вы можете обсуждать с пользователем или выполнять задачи, связанные с письмами. Письма требуют тему, адресата и тело письма.
Вы будете получать инструкции, начинающиеся с [Instruction], или ввод пользователя, начинающийся с [User]. Следуйте инструкциям.
"""

prompts = {
    'START': '[Instruction] Напишите WRITE_EMAIL, если пользователь хочет написать письмо, "QUESTION", если у пользователя есть конкретный вопрос, "OTHER" в любом другом случае. Напишите только одно слово.',
    'QUESTION': '[Instruction] Если вы можете ответить на вопрос, напишите "ANSWER", если вам нужно больше информации, напишите "MORE", если вы не можете ответить, напишите "OTHER". Напишите только одно слово.',
    'ANSWER': '[Instruction] Ответьте на вопрос пользователя.',
    'MORE': '[Instruction] Попросите пользователя предоставить больше информации в соответствии с предыдущими инструкциями.',
    'OTHER': '[Instruction] Дайте вежливый ответ или приветствие, если пользователь ведет вежливый разговор. В противном случае сообщите, что вы не можете ответить на вопрос или выполнить действие.',
    'WRITE_EMAIL': '[Instruction] Если отсутствует тема, адресат или тело письма, ответьте "MORE". В противном случае, если у вас есть вся информация, ответьте "ACTION_WRITE_EMAIL | subject:subject, recipient:recipient, message:message".',
    'ACTION_WRITE_EMAIL': '[Instruction] Письмо отправлено. Сообщите пользователю, что действие выполнено.'
}
actions = ['ACTION_WRITE_EMAIL']


class Discussion:
    """
    Класс, представляющий обсуждение с голосовым помощником.

    Атрибуты:
        state (str): Текущее состояние обсуждения.
        messages_history (list): История сообщений.
        stt_model: Модель распознавания речи Whisper.

    Методы:
        generate_answer: Генерирует ответ на основе сообщений.
        reset: Сбрасывает обсуждение в начальное состояние.
        do_action: Выполняет указанное действие.
        transcribe: Распознает текст из аудиофайла.
        discuss_from_audio: Начинает обсуждение на основе аудиофайла.
        discuss: Продолжает обсуждение на основе пользовательского ввода.
    """

    def __init__(self, state='START', messages_history=None) -> None:
        self.state = state
        self.messages_history = messages_history or [{'role': 'user', 'content': starting_prompt}]
        self.stt_model = whisper.load_model("base")

    def generate_answer(self, messages):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages
            )
            answer = response['choices'][0]['message']['content']
            usage = response['usage']  # Информация об использовании токенов
            print(f"\nОтвет: {answer}")
            print(f"Использование токенов: "
                  f"Промпт: {usage['prompt_tokens']}, "
                  f"Ответ: {usage['completion_tokens']}, "
                  f"Всего: {usage['total_tokens']}\n")
            return answer
        except openai.error.OpenAIError as e:
            logging.error(f"Ошибка OpenAI: {e}")
            return "Извините, произошла ошибка при обработке вашего запроса."

    def reset(self, start_state='START'):
        self.messages_history = [{'role': 'user', 'content': starting_prompt}]
        self.state = start_state
        self.previous_state = None

    def reset_to_previous_state(self):
        self.state = self.previous_state
        self.previous_state = None

    def to_state(self, state):
        self.previous_state = self.state
        self.state = state

    def do_action(self, action):
        """
        Выполняет указанное действие.

        Args:
            action (str): Действие для выполнения.
        """
        print(f'Выполняется действие: {action}')
        # Здесь можно добавить реальную логику для отправки письма или других действий.
        pass

    def transcribe(self, file):
        transcription = self.stt_model.transcribe(file)
        return transcription['text']

    def discuss_from_audio(self, file):
        if file:
            # Распознаем аудио и запускаем обсуждение с текстом
            return self.discuss(f'[User] {self.transcribe(file)}')
        # Если файл отсутствует
        return ''

    def discuss(self, input=None):
        if input is not None:
            self.messages_history.append({"role": "user", "content": input})

        # Генерация ответа
        completion = self.generate_answer(
            self.messages_history + [{"role": "user", "content": prompts[self.state]}]
        )

        # Проверка, является ли результат действием
        if completion.split("|")[0].strip() in actions:
            action = completion.split("|")[0].strip()
            self.to_state(action)
            self.do_action(completion)
            # Продолжаем обсуждение
            return self.discuss()
        # Проверка, является ли результат новым состоянием
        elif completion in prompts:
            self.to_state(completion)
            # Продолжаем обсуждение
            return self.discuss()
        # Ответ для пользователя
        else:
            self.messages_history.append({"role": "assistant", "content": completion})
            if self.state != 'MORE':
                # Возвращаемся в начальное состояние
                self.reset()
            else:
                # Возвращаемся к предыдущему состоянию
                self.reset_to_previous_state()
            return completion


if __name__ == '__main__':
    discussion = Discussion()

    gr.Interface(
        theme=gr.themes.Soft(),
        fn=discussion.discuss_from_audio,
        live=True,
        inputs=gr.Audio(type="filepath"),  # Убрали аргумент `source`
        outputs="text"
    ).launch()

