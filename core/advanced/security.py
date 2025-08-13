"""
MinionsAI v3.1 - Advanced Security & Authentication
Enterprise-grade security features including authentication, authorization, and access control.
"""

import hashlib
import secrets
import jwt
import bcrypt
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class UserRole(Enum):
    """User roles for access control."""
    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"
    API_USER = "api_user"


class Permission(Enum):
    """System permissions."""
    READ_CONVERSATIONS = "read_conversations"
    WRITE_CONVERSATIONS = "write_conversations"
    DELETE_CONVERSATIONS = "delete_conversations"
    MANAGE_USERS = "manage_users"
    MANAGE_MODELS = "manage_models"
    MANAGE_SYSTEM = "manage_system"
    API_ACCESS = "api_access"
    EXPORT_DATA = "export_data"
    VIEW_ANALYTICS = "view_analytics"
    MANAGE_SETTINGS = "manage_settings"


@dataclass
class User:
    """User account information."""
    username: str
    email: str
    password_hash: str
    role: UserRole
    permissions: Set[Permission] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    is_active: bool = True
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None
    api_key: Optional[str] = None
    session_tokens: Set[str] = field(default_factory=set)
    
    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """Convert user to dictionary."""
        data = {
            "username": self.username,
            "email": self.email,
            "role": self.role.value,
            "permissions": [p.value for p in self.permissions],
            "created_at": self.created_at.isoformat(),
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "is_active": self.is_active,
            "failed_login_attempts": self.failed_login_attempts,
            "locked_until": self.locked_until.isoformat() if self.locked_until else None
        }
        
        if include_sensitive:
            data.update({
                "password_hash": self.password_hash,
                "api_key": self.api_key,
                "session_tokens": list(self.session_tokens)
            })
        
        return data


@dataclass
class SecurityEvent:
    """Security event for audit logging."""
    timestamp: datetime = field(default_factory=datetime.now)
    event_type: str = ""
    username: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    success: bool = True
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "username": self.username,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "success": self.success,
            "details": self.details
        }


class SecurityManager:
    """
    Comprehensive security management system.
    """
    
    def __init__(self, secret_key: Optional[str] = None, data_dir: str = "data"):
        """
        Initialize the security manager.
        
        Args:
            secret_key: Secret key for JWT tokens
            data_dir: Directory for storing security data
        """
        self.secret_key = secret_key or self._generate_secret_key()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.users: Dict[str, User] = {}
        self.security_events: List[SecurityEvent] = []
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Security settings
        self.max_failed_attempts = 5
        self.lockout_duration = timedelta(minutes=30)
        self.session_timeout = timedelta(hours=24)
        self.password_min_length = 8
        self.require_complex_passwords = True
        
        # Load existing data
        self._load_users()
        self._create_default_admin()
    
    def _generate_secret_key(self) -> str:
        """Generate a secure secret key."""
        return secrets.token_urlsafe(32)
    
    def _load_users(self) -> None:
        """Load users from storage."""
        users_file = self.data_dir / "users.json"
        if users_file.exists():
            try:
                with open(users_file, 'r') as f:
                    users_data = json.load(f)
                
                for username, user_data in users_data.items():
                    user = User(
                        username=user_data["username"],
                        email=user_data["email"],
                        password_hash=user_data["password_hash"],
                        role=UserRole(user_data["role"]),
                        permissions=set(Permission(p) for p in user_data.get("permissions", [])),
                        created_at=datetime.fromisoformat(user_data["created_at"]),
                        last_login=datetime.fromisoformat(user_data["last_login"]) if user_data.get("last_login") else None,
                        is_active=user_data.get("is_active", True),
                        failed_login_attempts=user_data.get("failed_login_attempts", 0),
                        locked_until=datetime.fromisoformat(user_data["locked_until"]) if user_data.get("locked_until") else None,
                        api_key=user_data.get("api_key"),
                        session_tokens=set(user_data.get("session_tokens", []))
                    )
                    self.users[username] = user
                
                logger.info(f"Loaded {len(self.users)} users")
                
            except Exception as e:
                logger.error(f"Error loading users: {e}")
    
    def _save_users(self) -> None:
        """Save users to storage."""
        users_file = self.data_dir / "users.json"
        try:
            users_data = {
                username: user.to_dict(include_sensitive=True)
                for username, user in self.users.items()
            }
            
            with open(users_file, 'w') as f:
                json.dump(users_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving users: {e}")
    
    def _create_default_admin(self) -> None:
        """Create default admin user if none exists."""
        if not any(user.role == UserRole.ADMIN for user in self.users.values()):
            admin_password = secrets.token_urlsafe(12)
            
            success = self.create_user(
                username="admin",
                email="admin@minionsai.local",
                password=admin_password,
                role=UserRole.ADMIN
            )
            
            if success:
                logger.warning(f"Created default admin user with password: {admin_password}")
                logger.warning("Please change the admin password immediately!")
    
    def _hash_password(self, password: str) -> str:
        """Hash a password using bcrypt."""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify a password against its hash."""
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
    
    def _validate_password(self, password: str) -> Tuple[bool, List[str]]:
        """Validate password strength."""
        errors = []
        
        if len(password) < self.password_min_length:
            errors.append(f"Password must be at least {self.password_min_length} characters long")
        
        if self.require_complex_passwords:
            if not any(c.isupper() for c in password):
                errors.append("Password must contain at least one uppercase letter")
            if not any(c.islower() for c in password):
                errors.append("Password must contain at least one lowercase letter")
            if not any(c.isdigit() for c in password):
                errors.append("Password must contain at least one digit")
            if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
                errors.append("Password must contain at least one special character")
        
        return len(errors) == 0, errors
    
    def create_user(
        self,
        username: str,
        email: str,
        password: str,
        role: UserRole = UserRole.USER
    ) -> bool:
        """Create a new user."""
        if username in self.users:
            logger.warning(f"User {username} already exists")
            return False
        
        # Validate password
        is_valid, errors = self._validate_password(password)
        if not is_valid:
            logger.warning(f"Password validation failed: {', '.join(errors)}")
            return False
        
        # Create user
        password_hash = self._hash_password(password)
        permissions = self._get_default_permissions(role)
        
        user = User(
            username=username,
            email=email,
            password_hash=password_hash,
            role=role,
            permissions=permissions
        )
        
        # Generate API key for API users
        if role == UserRole.API_USER:
            user.api_key = self._generate_api_key()
        
        self.users[username] = user
        self._save_users()
        
        # Log security event
        self._log_security_event("user_created", username, success=True, details={"role": role.value})
        
        logger.info(f"Created user: {username} with role: {role.value}")
        return True
    
    def _get_default_permissions(self, role: UserRole) -> Set[Permission]:
        """Get default permissions for a role."""
        permission_map = {
            UserRole.ADMIN: {
                Permission.READ_CONVERSATIONS,
                Permission.WRITE_CONVERSATIONS,
                Permission.DELETE_CONVERSATIONS,
                Permission.MANAGE_USERS,
                Permission.MANAGE_MODELS,
                Permission.MANAGE_SYSTEM,
                Permission.API_ACCESS,
                Permission.EXPORT_DATA,
                Permission.VIEW_ANALYTICS,
                Permission.MANAGE_SETTINGS
            },
            UserRole.USER: {
                Permission.READ_CONVERSATIONS,
                Permission.WRITE_CONVERSATIONS,
                Permission.VIEW_ANALYTICS
            },
            UserRole.VIEWER: {
                Permission.READ_CONVERSATIONS,
                Permission.VIEW_ANALYTICS
            },
            UserRole.API_USER: {
                Permission.API_ACCESS,
                Permission.READ_CONVERSATIONS,
                Permission.WRITE_CONVERSATIONS
            }
        }
        
        return permission_map.get(role, set())
    
    def _generate_api_key(self) -> str:
        """Generate a secure API key."""
        return f"mai_{secrets.token_urlsafe(32)}"
    
    def authenticate_user(self, username: str, password: str, ip_address: Optional[str] = None) -> Optional[str]:
        """Authenticate a user and return a session token."""
        user = self.users.get(username)
        
        if not user or not user.is_active:
            self._log_security_event("login_failed", username, success=False, 
                                    ip_address=ip_address, details={"reason": "user_not_found"})
            return None
        
        # Check if user is locked
        if user.locked_until and datetime.now() < user.locked_until:
            self._log_security_event("login_failed", username, success=False,
                                    ip_address=ip_address, details={"reason": "account_locked"})
            return None
        
        # Verify password
        if not self._verify_password(password, user.password_hash):
            user.failed_login_attempts += 1
            
            # Lock account if too many failed attempts
            if user.failed_login_attempts >= self.max_failed_attempts:
                user.locked_until = datetime.now() + self.lockout_duration
                logger.warning(f"Account {username} locked due to failed login attempts")
            
            self._save_users()
            self._log_security_event("login_failed", username, success=False,
                                    ip_address=ip_address, details={"reason": "invalid_password"})
            return None
        
        # Successful login
        user.failed_login_attempts = 0
        user.locked_until = None
        user.last_login = datetime.now()
        
        # Generate session token
        session_token = self._generate_session_token(user)
        user.session_tokens.add(session_token)
        
        self._save_users()
        self._log_security_event("login_success", username, success=True, ip_address=ip_address)
        
        logger.info(f"User {username} authenticated successfully")
        return session_token
    
    def _generate_session_token(self, user: User) -> str:
        """Generate a JWT session token."""
        payload = {
            "username": user.username,
            "role": user.role.value,
            "permissions": [p.value for p in user.permissions],
            "exp": datetime.utcnow() + self.session_timeout,
            "iat": datetime.utcnow()
        }
        
        return jwt.encode(payload, self.secret_key, algorithm="HS256")
    
    def validate_session_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate a session token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            username = payload.get("username")
            
            # Check if user still exists and token is in active sessions
            user = self.users.get(username)
            if user and token in user.session_tokens:
                return payload
            
            return None
            
        except jwt.ExpiredSignatureError:
            logger.debug("Session token expired")
            return None
        except jwt.InvalidTokenError:
            logger.debug("Invalid session token")
            return None
    
    def logout_user(self, token: str) -> bool:
        """Logout a user by invalidating their session token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            username = payload.get("username")
            
            user = self.users.get(username)
            if user and token in user.session_tokens:
                user.session_tokens.remove(token)
                self._save_users()
                self._log_security_event("logout", username, success=True)
                return True
            
            return False
            
        except jwt.InvalidTokenError:
            return False
    
    def check_permission(self, token: str, required_permission: Permission) -> bool:
        """Check if a user has a specific permission."""
        payload = self.validate_session_token(token)
        if not payload:
            return False
        
        user_permissions = [Permission(p) for p in payload.get("permissions", [])]
        return required_permission in user_permissions
    
    def _log_security_event(
        self,
        event_type: str,
        username: Optional[str] = None,
        success: bool = True,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log a security event."""
        event = SecurityEvent(
            event_type=event_type,
            username=username,
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            details=details or {}
        )
        
        self.security_events.append(event)
        
        # Keep only recent events (last 1000)
        if len(self.security_events) > 1000:
            self.security_events = self.security_events[-1000:]
    
    def get_security_events(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent security events."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            event.to_dict() for event in self.security_events
            if event.timestamp > cutoff_time
        ]
    
    def get_user_info(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user information (without sensitive data)."""
        user = self.users.get(username)
        return user.to_dict() if user else None
    
    def list_users(self) -> List[Dict[str, Any]]:
        """List all users (without sensitive data)."""
        return [user.to_dict() for user in self.users.values()]
    
    def delete_user(self, username: str) -> bool:
        """Delete a user."""
        if username in self.users:
            del self.users[username]
            self._save_users()
            self._log_security_event("user_deleted", username, success=True)
            logger.info(f"Deleted user: {username}")
            return True
        return False
    
    def change_password(self, username: str, old_password: str, new_password: str) -> bool:
        """Change a user's password."""
        user = self.users.get(username)
        if not user:
            return False
        
        # Verify old password
        if not self._verify_password(old_password, user.password_hash):
            self._log_security_event("password_change_failed", username, success=False,
                                    details={"reason": "invalid_old_password"})
            return False
        
        # Validate new password
        is_valid, errors = self._validate_password(new_password)
        if not is_valid:
            logger.warning(f"New password validation failed: {', '.join(errors)}")
            return False
        
        # Update password
        user.password_hash = self._hash_password(new_password)
        
        # Invalidate all existing sessions
        user.session_tokens.clear()
        
        self._save_users()
        self._log_security_event("password_changed", username, success=True)
        
        logger.info(f"Password changed for user: {username}")
        return True


class AuthenticationManager:
    """
    Simplified authentication manager for basic use cases.
    """
    
    def __init__(self, security_manager: SecurityManager):
        """Initialize with a security manager."""
        self.security_manager = security_manager
    
    def login(self, username: str, password: str) -> Optional[str]:
        """Simple login method."""
        return self.security_manager.authenticate_user(username, password)
    
    def logout(self, token: str) -> bool:
        """Simple logout method."""
        return self.security_manager.logout_user(token)
    
    def is_authenticated(self, token: str) -> bool:
        """Check if token is valid."""
        return self.security_manager.validate_session_token(token) is not None
    
    def has_permission(self, token: str, permission: Permission) -> bool:
        """Check if user has permission."""
        return self.security_manager.check_permission(token, permission)
