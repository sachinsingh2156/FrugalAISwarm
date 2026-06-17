"""
Role-Based Access Control (RBAC) stubs.

In a production deployment, these stubs would be backed by an identity
provider (LDAP/Active Directory, OAuth2, or a local user store). For the
Phase-1 pilot, RBAC is enforced at the function level as a code pattern.

Roles
-----
researcher    : full read/write access to experiments, results, audit trail
educator      : can submit tasks, read own results; cannot access other users' data
admin         : full access including retention and RBAC management
readonly      : read-only access to anonymised results

Policy is DENY-by-default: any unrecognised role or action returns False.
"""
from __future__ import annotations

from enum import Enum
from functools import wraps
from typing import Callable, Any


class Role(str, Enum):
    RESEARCHER = "researcher"
    EDUCATOR   = "educator"
    ADMIN      = "admin"
    READONLY   = "readonly"


# ── Permission matrix ─────────────────────────────────────────────────────────
# (role, action) -> allowed: bool

_PERMISSIONS: dict[tuple[str, str], bool] = {
    # Researcher
    (Role.RESEARCHER, "submit_task"):       True,
    (Role.RESEARCHER, "read_results"):      True,
    (Role.RESEARCHER, "read_audit"):        True,
    (Role.RESEARCHER, "write_experiment"):  True,
    (Role.RESEARCHER, "read_all_results"):  True,
    (Role.RESEARCHER, "manage_retention"):  False,
    (Role.RESEARCHER, "manage_rbac"):       False,

    # Educator
    (Role.EDUCATOR, "submit_task"):         True,
    (Role.EDUCATOR, "read_results"):        True,   # own results only
    (Role.EDUCATOR, "read_audit"):          False,
    (Role.EDUCATOR, "write_experiment"):    False,
    (Role.EDUCATOR, "read_all_results"):    False,
    (Role.EDUCATOR, "manage_retention"):    False,
    (Role.EDUCATOR, "manage_rbac"):         False,

    # Admin
    (Role.ADMIN, "submit_task"):            True,
    (Role.ADMIN, "read_results"):           True,
    (Role.ADMIN, "read_audit"):             True,
    (Role.ADMIN, "write_experiment"):       True,
    (Role.ADMIN, "read_all_results"):       True,
    (Role.ADMIN, "manage_retention"):       True,
    (Role.ADMIN, "manage_rbac"):            True,

    # Read-only
    (Role.READONLY, "submit_task"):         False,
    (Role.READONLY, "read_results"):        True,   # anonymised only
    (Role.READONLY, "read_audit"):          False,
    (Role.READONLY, "write_experiment"):    False,
    (Role.READONLY, "read_all_results"):    False,
    (Role.READONLY, "manage_retention"):    False,
    (Role.READONLY, "manage_rbac"):         False,
}


def is_allowed(role: str | Role, action: str) -> bool:
    """Return True if *role* is permitted to perform *action*."""
    key = (Role(role) if isinstance(role, str) else role, action)
    return _PERMISSIONS.get(key, False)


def require_permission(action: str):
    """
    Decorator factory. Wrap a function to enforce RBAC at call time.

    Usage:
        @require_permission("read_audit")
        def view_audit_trail(role: str, ...):
            ...
    
    The decorated function must accept *role* as its first positional arg.
    Raises PermissionError if the role lacks the required permission.
    """
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(role: str | Role, *args: Any, **kwargs: Any) -> Any:
            if not is_allowed(role, action):
                raise PermissionError(
                    f"Role '{role}' does not have permission to '{action}'"
                )
            return fn(role, *args, **kwargs)
        return wrapper
    return decorator


# ── Pilot-mode shortcut ───────────────────────────────────────────────────────
# During Phase-1 pilot, all local runs are treated as RESEARCHER role.
# Override by setting SWARM_ROLE env variable.

import os

def get_current_role() -> Role:
    """Return the effective role for the current process."""
    raw = os.getenv("SWARM_ROLE", Role.RESEARCHER.value)
    try:
        return Role(raw)
    except ValueError:
        return Role.READONLY
