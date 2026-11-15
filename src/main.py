from VanillaAgent import VanillaAgent
import sys

class Main():
    def main():
        chenyiGpt = VanillaAgent(tempature = 0.8, max_length = 256, memory_length = 256)
        COMMANDS = {
        "/next": chenyiGpt.next_sentence,
        "r": chenyiGpt.reset,
        }

        while True:
                user_input = input("\nYou: ")

                if user_input == "q":
                    break

                elif user_input in COMMANDS:
                    print(COMMANDS[user_input]())

                elif user_input.strip():
                    print(chenyiGpt.complete(user_input))
                
    if __name__ == "__main__":
        main()