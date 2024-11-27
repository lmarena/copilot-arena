import base64
import pytest
import bcrypt
from firebase_admin import firestore
from unittest.mock import Mock, patch
from src.user_repository import (
    UserRepository,
    UserNotFoundError,
    PasswordIncorrectError,
    UsernameExistsError,
)


@pytest.fixture
def user_repo():
    with patch("firebase_admin.firestore.client") as mock_firestore:
        mock_collection = Mock()
        mock_firestore.return_value.collection.return_value = mock_collection
        yield UserRepository()


def test_get_user_by_id(user_repo):
    mock_query = user_repo.collection_ref.where.return_value
    mock_query.get.return_value = [
        Mock(to_dict=lambda: {"userId": "123", "username": "testuser"})
    ]

    user = user_repo.get_user_by_id("123")
    assert user["userId"] == "123"
    assert user["username"] == "testuser"


def test_get_user_by_id_not_found(user_repo):
    mock_query = user_repo.collection_ref.where.return_value
    mock_query.get.return_value = []

    with pytest.raises(UserNotFoundError):
        user_repo.get_user_by_id("nonexistent")


def test_get_user_by_username(user_repo):
    mock_query = user_repo.collection_ref.where.return_value
    mock_query.get.return_value = [
        Mock(to_dict=lambda: {"userId": "123", "username": "testuser"})
    ]

    user = user_repo.get_user_by_username("testuser")
    assert user["userId"] == "123"
    assert user["username"] == "testuser"


def test_get_user_by_username_not_found(user_repo):
    mock_query = user_repo.collection_ref.where.return_value
    mock_query.get.return_value = []

    with pytest.raises(UserNotFoundError):
        user_repo.get_user_by_username("nonexistent")


def test_authenticate_success(user_repo):
    mock_query = user_repo.collection_ref.where.return_value
    password = "password123"
    password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    password_hash = base64.b64encode(password_hash).decode("utf-8")
    mock_query.get.return_value = [
        Mock(
            to_dict=lambda: {
                "userId": "123",
                "username": "testuser",
                "passwordHash": password_hash,
            }
        )
    ]

    user = user_repo.authenticate("testuser", "123", password)
    assert user["userId"] == "123"
    assert user["username"] == "testuser"


def test_authenticate_incorrect_password(user_repo):
    mock_query = user_repo.collection_ref.where.return_value
    mock_query.get.return_value = [
        Mock(
            to_dict=lambda: {
                "userId": "123",
                "username": "testuser",
                "passwordHash": "JDJiJDEyJC9WRVZkMDRwcmZXUlVOYXpSLlVXenVUUHNFWFhkNUxrN0xwRXg4WnNrWjNaV2ZTd1JUMU9P",
            }
        )
    ]

    with pytest.raises(PasswordIncorrectError):
        user_repo.authenticate("testuser", "wrongpassword")


def test_create_user(user_repo):
    mock_query = user_repo.collection_ref.where.return_value
    mock_query.get.return_value = []

    user_repo.create_user("123", "newuser", "password123")
    user_repo.collection_ref.add.assert_called_once()


def test_create_user_existing_username(user_repo):
    mock_query = user_repo.collection_ref.where.return_value
    mock_query.get.return_value = [Mock()]

    with pytest.raises(UsernameExistsError):
        user_repo.create_user("123", "existinguser", "password123")


def test_update_user(user_repo):
    mock_query = user_repo.collection_ref.where.return_value
    mock_query.get.return_value = [Mock(id="doc123")]

    with patch.object(UserRepository, "authenticate"):
        user_repo.update_user("testuser", "oldpassword", "newpassword")

    user_repo.collection_ref.document.assert_called_with("doc123")
    user_repo.collection_ref.document.return_value.update.assert_called_once()


def test_delete_user(user_repo):
    mock_doc = Mock()
    mock_doc.get.return_value.exists = True
    user_repo.collection_ref.document.return_value = mock_doc

    user_repo.delete_user("123")
    mock_doc.delete.assert_called_once()


def test_delete_user_not_found(user_repo):
    mock_doc = Mock()
    mock_doc.get.return_value.exists = False
    user_repo.collection_ref.document.return_value = mock_doc

    with pytest.raises(UserNotFoundError):
        user_repo.delete_user("nonexistent")
