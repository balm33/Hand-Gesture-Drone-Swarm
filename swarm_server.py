import socket
import threading
import time
import sys

# server settings
SERVER_HOST = '0.0.0.0'
SERVER_PORT = 12345

# shared state
movement_command = None
lock = threading.Lock()
server_socket = None
input_clients = []
drone_clients = []
running = True

command_queue = []

def process_command(command):
    if command == "open_hand":
        return None
    elif command == "index_up":
        return "MOVE_UP"
    elif command == "closed_fist":
        return None
    elif command == "peace_sign":
        return "MOVE_DOWN"
    elif command == "pinky_up":
        return "ROTATE_RIGHT"
    elif command == "two_up":
        return "MOVE_FORWARD"
    elif command == "three_up":
        return "MOVE_BACK"
    elif command == "thumb_side":
        return "ROTATE_LEFT"
    elif command == "exit":
        return "exit"
    else:
        return None

# handle incoming client connection
def handle_client(client_socket, address):
    global movement_command
    print(f'[NEW CONNECTION] {address} connected')
    role_data = client_socket.recv(1024).decode('utf-8')
    role = role_data.strip()

    if role == "input":
        print("Input client connected")
        input_clients.append(client_socket)

        try:
            while running:
                # receive command from input client
                command = client_socket.recv(1024).decode('utf-8')
                if command:
                    print(f'Received command from input client: {command}')

                    # add command to queue to send to drones
                    command_queue.append(process_command(command))
        except Exception as e:
            print(f"Error listening to input client: {e}")
            client_socket.close()

    elif role == "drone":
        print("Drone client connected")
        drone_clients.append(client_socket)
        
        try:
            while running:
                # receive location data



                # send movement command if available
                with lock:
                    if movement_command:
                        client_socket.sendall(movement_command.encode('utf-8'))
                        movement_command = None
                time.sleep(0.05)  # Important: avoid CPU overload, send one at a time
        except (BrokenPipeError, ConnectionResetError):
            print(f"[DISCONNECTED] {address} forcibly closed.")
        except Exception as e:
            print(f"[ERROR] {address} error: {e}")
        finally:
            try:
                client_socket.close()
            except:
                pass

    else:
        print(f"Unknown role from {address}: {role}")
        client_socket.close()


def shutdown_server():
    global running
    print("[SHUTDOWN] Closing server and all connections")

    running = False

    # close client sockets
    for client in drone_clients:
        try:
            client.shutdown(socket.SHUT_RDWR)
            client.close()
        except:
            pass
    for client in input_clients:
        try:
            client.shutdown(socket.SHUT_RDWR)
            client.close()
        except:
            pass
    
    # close server socket
    try:
        server_socket.close()
    except:
        pass

    print("[SHUTDOWN COMPLETE]")
    sys.exit(0)

# main server function
def start_server():
    global movement_command

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # reuse port immediately
    server_socket.bind((SERVER_HOST, SERVER_PORT))
    server_socket.listen(5)
    print(f'[LISTENING] Server is listening on {SERVER_HOST}:{SERVER_PORT}')

    # thread to accept new clients
    def accept_clients():
        while running:
            try:
                client_socket, address = server_socket.accept()
                thread = threading.Thread(target=handle_client, args=(client_socket, address))
                thread.start()
            except:
                break
    
    threading.Thread(target=accept_clients, daemon=True).start()

    # main thread handles operator inputs
    new_command = None
    while True:
        try:
            # new_command = input("").strip()
            if command_queue:
                new_command = command_queue[0]
                command_queue.pop(0)

            if new_command and new_command.lower() == "exit":
                shutdown_server()
                break

            with lock:
                movement_command = new_command
        except KeyboardInterrupt:
            shutdown_server()

if __name__ == "__main__":
    start_server()