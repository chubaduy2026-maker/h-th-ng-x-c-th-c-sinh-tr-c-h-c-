from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
from bson import ObjectId
from bson.errors import InvalidId
from pymongo import MongoClient, ReturnDocument, UpdateOne
from pymongo.collection import Collection

try:
    from src.config import ATTENDANCE_LOG_COLLECTION, COLLECTION_NAME, DB_NAME, MONGO_URI, validate_config
except ImportError:
    from config import ATTENDANCE_LOG_COLLECTION, COLLECTION_NAME, DB_NAME, MONGO_URI, validate_config


_client: MongoClient | None = None
_collection: Collection | None = None
_logs_collection: Collection | None = None


def _get_collection() -> Collection:
    global _client, _collection

    if _collection is not None:
        return _collection

    validate_config()

    _client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    database = _client[DB_NAME]
    _collection = database[COLLECTION_NAME]

    # Ensure one student ID maps to exactly one document.
    _collection.create_index("mssv", unique=True)
    return _collection


def _get_logs_collection() -> Collection:
    global _logs_collection

    if _logs_collection is not None:
        return _logs_collection

    # Ensure client is initialized and config validated.
    _get_collection()

    if _client is None:
        raise RuntimeError("MongoDB client is not initialized.")

    _logs_collection = _client[DB_NAME][ATTENDANCE_LOG_COLLECTION]
    _logs_collection.create_index("timestamp")
    _logs_collection.create_index([("mssv", 1), ("timestamp", -1)])
    return _logs_collection


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        raise ValueError("Embedding has zero norm and cannot be normalized.")
    return vector / norm


def register_student(mssv: str, name: str, vector: np.ndarray | list[float]) -> dict[str, Any]:
    if not mssv or not mssv.strip():
        raise ValueError("mssv is required.")
    if not name or not name.strip():
        raise ValueError("name is required.")

    embedding = np.asarray(vector, dtype=np.float32).flatten()
    if embedding.shape[0] != 512:
        raise ValueError(f"Embedding size must be 512, got {embedding.shape[0]}.")

    embedding = _normalize_vector(embedding)
    now = datetime.now(timezone.utc)

    collection = _get_collection()
    document = collection.find_one_and_update(
        {"mssv": mssv.strip()},
        {
            "$set": {
                "mssv": mssv.strip(),
                "name": name.strip(),
                "embedding": embedding.tolist(),
                "is_present": False,
                "updated_at": now,
            },
            "$setOnInsert": {
                "created_at": now,
            },
        },
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )

    if document is None:
        raise RuntimeError("Failed to register student.")

    return {
        "id": str(document.get("_id")),
        "mssv": document.get("mssv"),
        "name": document.get("name"),
        "is_present": bool(document.get("is_present", False)),
    }


def add_student(mssv: str, name: str, face_vector: np.ndarray | list[float]) -> dict[str, Any]:
    # Keep a dedicated API name for the web app layer.
    return register_student(mssv=mssv, name=name, vector=face_vector)


def get_all_students() -> list[dict[str, Any]]:
    collection = _get_collection()
    cursor = collection.find(
        {},
        {
            "mssv": 1,
            "name": 1,
            "is_present": 1,
            "created_at": 1,
            "updated_at": 1,
            "last_attended_at": 1,
        },
    ).sort("mssv", 1)

    students: list[dict[str, Any]] = []
    for doc in cursor:
        students.append(
            {
                "id": str(doc.get("_id")),
                "mssv": str(doc.get("mssv", "")).strip(),
                "name": str(doc.get("name", "")).strip(),
                "is_present": bool(doc.get("is_present", False)),
                "created_at": doc.get("created_at"),
                "updated_at": doc.get("updated_at"),
                "last_attended_at": doc.get("last_attended_at"),
            }
        )

    return students


def delete_student(mssv: str) -> dict[str, Any]:
    clean_mssv = str(mssv).strip()
    if not clean_mssv:
        raise ValueError("mssv is required.")

    collection = _get_collection()
    result = collection.delete_one({"mssv": clean_mssv})

    return {
        "mssv": clean_mssv,
        "deleted": int(result.deleted_count),
    }


def add_attendance_log(mssv: str, name: str, timestamp: datetime | None = None) -> dict[str, Any]:
    clean_mssv = str(mssv).strip()
    clean_name = str(name).strip()

    if not clean_mssv:
        raise ValueError("mssv is required.")
    if not clean_name:
        raise ValueError("name is required.")

    log_time = timestamp or datetime.now(timezone.utc)
    logs_collection = _get_logs_collection()

    document = {
        "mssv": clean_mssv,
        "name": clean_name,
        "timestamp": log_time,
    }
    insert_result = logs_collection.insert_one(document)

    return {
        "id": str(insert_result.inserted_id),
        "mssv": clean_mssv,
        "name": clean_name,
        "timestamp": log_time,
    }


def get_attendance_logs(limit: int = 300) -> list[dict[str, Any]]:
    logs_collection = _get_logs_collection()
    safe_limit = max(1, min(int(limit), 2000))

    cursor = logs_collection.find(
        {},
        {
            "_id": 1,
            "mssv": 1,
            "name": 1,
            "timestamp": 1,
        },
    ).sort("timestamp", -1).limit(safe_limit)

    logs: list[dict[str, Any]] = []
    for item in cursor:
        logs.append(
            {
                "id": str(item.get("_id")),
                "mssv": str(item.get("mssv", "")).strip(),
                "name": str(item.get("name", "")).strip(),
                "timestamp": item.get("timestamp"),
            }
        )

    return logs


def delete_attendance_log(log_id: str) -> dict[str, Any]:
    clean_id = str(log_id).strip()
    if not clean_id:
        raise ValueError("log_id is required.")

    try:
        object_id = ObjectId(clean_id)
    except InvalidId as exc:
        raise ValueError("Invalid log id format.") from exc

    logs_collection = _get_logs_collection()
    result = logs_collection.delete_one({"_id": object_id})

    return {
        "log_id": clean_id,
        "deleted": int(result.deleted_count),
    }


def delete_unknown_attendance_logs() -> dict[str, Any]:
    logs_collection = _get_logs_collection()
    result = logs_collection.delete_many(
        {
            "$or": [
                {"mssv": {"$in": ["UNKNOWN", "", None]}},
                {"name": {"$in": ["Unknown", "", None]}},
            ]
        }
    )

    return {
        "deleted": int(result.deleted_count),
    }


def update_attendance(mssv: str) -> dict[str, Any]:
    clean_mssv = str(mssv).strip()
    if not clean_mssv:
        raise ValueError("mssv is required.")

    collection = _get_collection()
    now = datetime.now(timezone.utc)

    document = collection.find_one_and_update(
        {"mssv": clean_mssv},
        {
            "$set": {
                "is_present": True,
                "last_attended_at": now,
                "updated_at": now,
            }
        },
        return_document=ReturnDocument.AFTER,
    )

    if document is None:
        raise ValueError(f"Student not found: {clean_mssv}")

    return {
        "id": str(document.get("_id")),
        "mssv": str(document.get("mssv", "")).strip(),
        "name": str(document.get("name", "")).strip(),
        "is_present": bool(document.get("is_present", False)),
        "updated_at": document.get("updated_at"),
        "last_attended_at": document.get("last_attended_at"),
    }


def pull_data() -> list[dict[str, Any]]:
    collection = _get_collection()

    records: list[dict[str, Any]] = []
    cursor = collection.find(
        {"embedding": {"$exists": True}},
        {
            "_id": 0,
            "mssv": 1,
            "name": 1,
            "embedding": 1,
            "is_present": 1,
        },
    )

    for doc in cursor:
        raw_embedding = np.asarray(doc.get("embedding", []), dtype=np.float32).flatten()
        if raw_embedding.shape[0] != 512:
            continue

        try:
            embedding = _normalize_vector(raw_embedding)
        except ValueError:
            continue

        records.append(
            {
                "mssv": str(doc.get("mssv", "")).strip(),
                "name": str(doc.get("name", "")).strip(),
                "embedding": embedding,
                "is_present": bool(doc.get("is_present", False)),
            }
        )

    return [r for r in records if r["mssv"]]


def sync_attendance(local_attendance: dict[str, dict[str, Any]]) -> dict[str, int]:
    if not local_attendance:
        return {"matched": 0, "modified": 0}

    collection = _get_collection()
    now = datetime.now(timezone.utc)

    operations: list[UpdateOne] = []
    for mssv, info in local_attendance.items():
        clean_mssv = str(mssv).strip()
        if not clean_mssv:
            continue

        is_present = bool(info.get("is_present", False))
        update_doc: dict[str, Any] = {
            "is_present": is_present,
            "updated_at": now,
        }
        if is_present:
            update_doc["last_attended_at"] = now

        operations.append(
            UpdateOne(
                {"mssv": clean_mssv},
                {"$set": update_doc},
                upsert=False,
            )
        )

    if not operations:
        return {"matched": 0, "modified": 0}

    result = collection.bulk_write(operations, ordered=False)
    return {
        "matched": int(result.matched_count),
        "modified": int(result.modified_count),
    }


def reset_all_attendance() -> dict[str, int]:
    collection = _get_collection()
    now = datetime.now(timezone.utc)

    result = collection.update_many(
        {},
        {
            "$set": {
                "is_present": False,
                "updated_at": now,
            }
        },
    )

    return {
        "matched": int(result.matched_count),
        "modified": int(result.modified_count),
    }
