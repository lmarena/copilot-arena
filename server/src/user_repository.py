import base64
from firebase_admin import firestore
import bcrypt


# Custom Errors
class UserNotFoundError(Exception):
    pass


class PasswordIncorrectError(Exception):
    pass


class UsernameExistsError(Exception):
    pass


class UserRepository:
    # Constants for database keys
    USER_ID_KEY = "userId"
    USERNAME_KEY = "username"
    PASSWORD_HASH_KEY = "passwordHash"
    METADATA_KEY = "metadata"

    def __init__(self):
        self.db = firestore.client()
        collection_name = "users"
        self.collection_ref = self.db.collection(collection_name)
        self.function_call_count = {
            "get_user_by_id": 0,
            "get_user_by_username": 0,
            "authenticate": 0,
            "create_user": 0,
            "update_user": 0,
            "delete_user": 0,
        }

    def track_function_call(self, function_name):
        self.function_call_count[function_name] += 1
        print(self.function_call_count)

    def get_user_by_id(self, user_id: str):
        self.track_function_call("get_user_by_id")
        query = self.collection_ref.where(self.USER_ID_KEY, "==", user_id)
        results = query.get()
        if not results:
            raise UserNotFoundError(f"User with ID {user_id} not found")
        return results[0].to_dict()

    def get_user_by_username(self, username: str):
        self.track_function_call("get_user_by_username")
        query = self.collection_ref.where(self.USERNAME_KEY, "==", username)
        results = query.get()
        if not results:
            raise UserNotFoundError(f"User with username {username} not found")
        return results[0].to_dict()

    def authenticate(self, username: str, user_id: str, password: str = None):
        self.track_function_call("authenticate")
        try:
            user = self.get_user_by_id(user_id)
        except:
            user = self.get_user_by_username(username)

        if self.PASSWORD_HASH_KEY not in user.keys():
            return user

        if password is None:
            raise PasswordIncorrectError("Invalid username or password")

        # Decode the base64 string back to bytes
        stored_hash = base64.b64decode(user[self.PASSWORD_HASH_KEY])
        if not bcrypt.checkpw(password.encode("utf-8"), stored_hash):
            raise PasswordIncorrectError("Invalid username or password")
        return user

    def create_user(
        self, user_id: str, username: str, password: str, metadata: dict = None
    ):
        self.track_function_call("create_user")
        try:
            self.get_user_by_username(username)
            raise UsernameExistsError("Username already exists")
        except UserNotFoundError:
            pass
        user_data = {
            self.USER_ID_KEY: user_id,
            self.USERNAME_KEY: username,
        }
        if password is not None:
            password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
            # Convert byte string to base64-encoded string
            user_data[self.PASSWORD_HASH_KEY] = base64.b64encode(password_hash).decode(
                "utf-8"
            )
        if metadata is not None:
            user_data[self.METADATA_KEY] = metadata
        self.collection_ref.add(user_data)

    def update_user(
        self,
        username: str = None,
        old_password: str = None,
        new_password: str = None,
        metadata: dict = None,
    ):
        self.track_function_call("update_user")
        user_doc = self.collection_ref.where(self.USERNAME_KEY, "==", username).get()
        if not user_doc:
            raise UserNotFoundError(f"User with username {username} not found")
        doc_id = user_doc[0].id
        update_data = {}
        self.authenticate(username, old_password)
        if new_password:
            password_hash = bcrypt.hashpw(
                new_password.encode("utf-8"), bcrypt.gensalt()
            )
            # Convert byte string to base64-encoded string
            update_data[self.PASSWORD_HASH_KEY] = base64.b64encode(
                password_hash
            ).decode("utf-8")
        if metadata:
            update_data[self.METADATA_KEY] = metadata
        self.collection_ref.document(doc_id).update(update_data)

    def delete_user(self, user_id: str):
        self.track_function_call("delete_user")
        doc_ref = self.collection_ref.document(user_id)
        if not doc_ref.get().exists:
            raise UserNotFoundError(f"User with ID {user_id} not found")
        doc_ref.delete()
