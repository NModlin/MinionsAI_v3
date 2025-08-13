"""
MinionsAI v3.1 - Advanced Model Management
Multi-model support, optimization, and intelligent model selection.
"""

import asyncio
import time
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import statistics

from langchain_ollama import OllamaLLM
from langchain_core.messages import BaseMessage
import requests

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Supported model types."""
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"


class ModelCapability(Enum):
    """Model capabilities."""
    TEXT_GENERATION = "text_generation"
    CODE_GENERATION = "code_generation"
    ANALYSIS = "analysis"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    REASONING = "reasoning"
    MATH = "mathematics"
    CREATIVE_WRITING = "creative_writing"


@dataclass
class ModelInfo:
    """Model information and metadata."""
    name: str
    model_type: ModelType
    capabilities: List[ModelCapability] = field(default_factory=list)
    context_length: int = 4096
    parameters: Optional[str] = None
    description: str = ""
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    is_available: bool = False
    performance_score: float = 0.0
    avg_response_time: float = 0.0
    success_rate: float = 1.0
    cost_per_token: float = 0.0
    last_used: Optional[datetime] = None
    usage_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "model_type": self.model_type.value,
            "capabilities": [cap.value for cap in self.capabilities],
            "context_length": self.context_length,
            "parameters": self.parameters,
            "description": self.description,
            "is_available": self.is_available,
            "performance_score": self.performance_score,
            "avg_response_time": self.avg_response_time,
            "success_rate": self.success_rate,
            "cost_per_token": self.cost_per_token,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "usage_count": self.usage_count
        }


@dataclass
class ModelPerformanceMetrics:
    """Performance metrics for a model."""
    model_name: str
    timestamp: datetime = field(default_factory=datetime.now)
    response_time: float = 0.0
    tokens_generated: int = 0
    success: bool = True
    error_message: Optional[str] = None
    task_type: str = "general"
    quality_score: Optional[float] = None


class ModelManager:
    """
    Advanced model management with multi-model support and intelligent selection.
    """
    
    def __init__(self):
        """Initialize the model manager."""
        self.models: Dict[str, ModelInfo] = {}
        self.model_instances: Dict[str, Any] = {}
        self.performance_history: List[ModelPerformanceMetrics] = []
        self.default_model: Optional[str] = None
        self.model_selection_strategy = "performance"  # performance, cost, speed
        self._initialization_lock = asyncio.Lock()
    
    def register_model(
        self,
        name: str,
        model_type: ModelType,
        capabilities: List[ModelCapability],
        context_length: int = 4096,
        parameters: Optional[str] = None,
        description: str = "",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        cost_per_token: float = 0.0
    ) -> None:
        """Register a new model."""
        model_info = ModelInfo(
            name=name,
            model_type=model_type,
            capabilities=capabilities,
            context_length=context_length,
            parameters=parameters,
            description=description,
            base_url=base_url,
            api_key=api_key,
            cost_per_token=cost_per_token
        )
        
        self.models[name] = model_info
        logger.info(f"Registered model: {name} ({model_type.value})")
    
    async def initialize_models(self) -> Dict[str, bool]:
        """Initialize all registered models."""
        async with self._initialization_lock:
            results = {}
            
            for model_name, model_info in self.models.items():
                try:
                    success = await self._initialize_single_model(model_name, model_info)
                    results[model_name] = success
                    model_info.is_available = success
                    
                    if success and self.default_model is None:
                        self.default_model = model_name
                        
                except Exception as e:
                    logger.error(f"Error initializing model {model_name}: {e}")
                    results[model_name] = False
                    model_info.is_available = False
            
            logger.info(f"Model initialization complete. Available: {sum(results.values())}/{len(results)}")
            return results
    
    async def _initialize_single_model(self, model_name: str, model_info: ModelInfo) -> bool:
        """Initialize a single model."""
        try:
            if model_info.model_type == ModelType.OLLAMA:
                # Check if Ollama model is available
                if await self._check_ollama_model(model_name, model_info.base_url):
                    llm = OllamaLLM(
                        model=model_name,
                        base_url=model_info.base_url or "http://localhost:11434"
                    )
                    self.model_instances[model_name] = llm
                    return True
            
            elif model_info.model_type == ModelType.OPENAI:
                # Initialize OpenAI model (placeholder)
                logger.info(f"OpenAI model {model_name} registered (implementation needed)")
                return False
            
            elif model_info.model_type == ModelType.ANTHROPIC:
                # Initialize Anthropic model (placeholder)
                logger.info(f"Anthropic model {model_name} registered (implementation needed)")
                return False
            
            return False
            
        except Exception as e:
            logger.error(f"Error initializing {model_name}: {e}")
            return False
    
    async def _check_ollama_model(self, model_name: str, base_url: Optional[str] = None) -> bool:
        """Check if Ollama model is available."""
        try:
            url = (base_url or "http://localhost:11434") + "/api/tags"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])
                return any(model.get("name", "").startswith(model_name) for model in models)
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking Ollama model {model_name}: {e}")
            return False
    
    def get_best_model_for_task(
        self,
        task_type: str,
        required_capabilities: Optional[List[ModelCapability]] = None,
        max_response_time: Optional[float] = None,
        max_cost: Optional[float] = None
    ) -> Optional[str]:
        """Select the best model for a specific task."""
        available_models = [
            name for name, info in self.models.items() 
            if info.is_available
        ]
        
        if not available_models:
            return self.default_model
        
        # Filter by capabilities
        if required_capabilities:
            available_models = [
                name for name in available_models
                if all(cap in self.models[name].capabilities for cap in required_capabilities)
            ]
        
        if not available_models:
            return self.default_model
        
        # Filter by constraints
        if max_response_time:
            available_models = [
                name for name in available_models
                if self.models[name].avg_response_time <= max_response_time
            ]
        
        if max_cost:
            available_models = [
                name for name in available_models
                if self.models[name].cost_per_token <= max_cost
            ]
        
        if not available_models:
            return self.default_model
        
        # Select based on strategy
        if self.model_selection_strategy == "performance":
            return max(available_models, key=lambda name: self.models[name].performance_score)
        elif self.model_selection_strategy == "speed":
            return min(available_models, key=lambda name: self.models[name].avg_response_time)
        elif self.model_selection_strategy == "cost":
            return min(available_models, key=lambda name: self.models[name].cost_per_token)
        
        return available_models[0]
    
    async def generate_response(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        task_type: str = "general",
        **kwargs
    ) -> Tuple[str, ModelPerformanceMetrics]:
        """Generate response using specified or best model."""
        if model_name is None:
            model_name = self.get_best_model_for_task(task_type)
        
        if model_name is None or model_name not in self.model_instances:
            raise ValueError(f"Model {model_name} not available")
        
        model_instance = self.model_instances[model_name]
        model_info = self.models[model_name]
        
        # Record performance metrics
        start_time = time.time()
        success = True
        error_message = None
        response = ""
        
        try:
            # Generate response
            if hasattr(model_instance, 'invoke'):
                response = model_instance.invoke(prompt, **kwargs)
                if hasattr(response, 'content'):
                    response = response.content
                else:
                    response = str(response)
            else:
                response = await model_instance.agenerate([prompt], **kwargs)
            
        except Exception as e:
            success = False
            error_message = str(e)
            logger.error(f"Error generating response with {model_name}: {e}")
            raise
        
        finally:
            # Record metrics
            response_time = time.time() - start_time
            tokens_generated = len(response.split()) if response else 0
            
            metrics = ModelPerformanceMetrics(
                model_name=model_name,
                response_time=response_time,
                tokens_generated=tokens_generated,
                success=success,
                error_message=error_message,
                task_type=task_type
            )
            
            self.performance_history.append(metrics)
            self._update_model_metrics(model_name, metrics)
        
        return response, metrics
    
    def _update_model_metrics(self, model_name: str, metrics: ModelPerformanceMetrics) -> None:
        """Update model performance metrics."""
        model_info = self.models[model_name]
        
        # Update usage stats
        model_info.usage_count += 1
        model_info.last_used = datetime.now()
        
        # Update performance metrics
        recent_metrics = [
            m for m in self.performance_history
            if m.model_name == model_name and 
            m.timestamp > datetime.now() - timedelta(hours=24)
        ]
        
        if recent_metrics:
            # Calculate averages
            response_times = [m.response_time for m in recent_metrics]
            model_info.avg_response_time = statistics.mean(response_times)
            
            success_count = sum(1 for m in recent_metrics if m.success)
            model_info.success_rate = success_count / len(recent_metrics)
            
            # Calculate performance score (higher is better)
            # Factors: success rate, speed (inverted), usage frequency
            speed_score = 1.0 / (model_info.avg_response_time + 0.1)  # Avoid division by zero
            usage_score = min(model_info.usage_count / 100.0, 1.0)  # Normalize usage
            
            model_info.performance_score = (
                model_info.success_rate * 0.5 +
                speed_score * 0.3 +
                usage_score * 0.2
            )
    
    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get information about a specific model."""
        return self.models.get(model_name)
    
    def get_available_models(self) -> List[str]:
        """Get list of available model names."""
        return [name for name, info in self.models.items() if info.is_available]
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get comprehensive model statistics."""
        total_models = len(self.models)
        available_models = len(self.get_available_models())
        
        # Performance stats
        recent_metrics = [
            m for m in self.performance_history
            if m.timestamp > datetime.now() - timedelta(hours=24)
        ]
        
        avg_response_time = 0.0
        success_rate = 1.0
        
        if recent_metrics:
            avg_response_time = statistics.mean([m.response_time for m in recent_metrics])
            success_count = sum(1 for m in recent_metrics if m.success)
            success_rate = success_count / len(recent_metrics)
        
        return {
            "total_models": total_models,
            "available_models": available_models,
            "default_model": self.default_model,
            "selection_strategy": self.model_selection_strategy,
            "avg_response_time": avg_response_time,
            "success_rate": success_rate,
            "total_requests": len(self.performance_history),
            "recent_requests": len(recent_metrics)
        }
    
    def set_default_model(self, model_name: str) -> bool:
        """Set the default model."""
        if model_name in self.models and self.models[model_name].is_available:
            self.default_model = model_name
            logger.info(f"Default model set to: {model_name}")
            return True
        return False
    
    def set_selection_strategy(self, strategy: str) -> bool:
        """Set model selection strategy."""
        valid_strategies = ["performance", "speed", "cost"]
        if strategy in valid_strategies:
            self.model_selection_strategy = strategy
            logger.info(f"Model selection strategy set to: {strategy}")
            return True
        return False


class ModelOptimizer:
    """
    Model performance optimization and fine-tuning capabilities.
    """
    
    def __init__(self, model_manager: ModelManager):
        """Initialize the model optimizer."""
        self.model_manager = model_manager
        self.optimization_history: List[Dict[str, Any]] = []
    
    def analyze_model_performance(self, model_name: str, days: int = 7) -> Dict[str, Any]:
        """Analyze model performance over time."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        model_metrics = [
            m for m in self.model_manager.performance_history
            if m.model_name == model_name and m.timestamp > cutoff_date
        ]
        
        if not model_metrics:
            return {"error": f"No performance data for {model_name}"}
        
        # Calculate statistics
        response_times = [m.response_time for m in model_metrics]
        success_count = sum(1 for m in model_metrics if m.success)
        
        analysis = {
            "model_name": model_name,
            "analysis_period_days": days,
            "total_requests": len(model_metrics),
            "success_rate": success_count / len(model_metrics),
            "avg_response_time": statistics.mean(response_times),
            "median_response_time": statistics.median(response_times),
            "min_response_time": min(response_times),
            "max_response_time": max(response_times),
            "response_time_std": statistics.stdev(response_times) if len(response_times) > 1 else 0
        }
        
        # Performance trends
        if len(model_metrics) > 10:
            recent_half = model_metrics[len(model_metrics)//2:]
            older_half = model_metrics[:len(model_metrics)//2]
            
            recent_avg = statistics.mean([m.response_time for m in recent_half])
            older_avg = statistics.mean([m.response_time for m in older_half])
            
            analysis["performance_trend"] = "improving" if recent_avg < older_avg else "declining"
            analysis["trend_magnitude"] = abs(recent_avg - older_avg) / older_avg
        
        return analysis
    
    def recommend_optimizations(self, model_name: str) -> List[Dict[str, Any]]:
        """Recommend optimizations for a model."""
        analysis = self.analyze_model_performance(model_name)
        recommendations = []
        
        if "error" in analysis:
            return recommendations
        
        # Response time recommendations
        if analysis["avg_response_time"] > 10.0:
            recommendations.append({
                "type": "performance",
                "priority": "high",
                "issue": "High response time",
                "recommendation": "Consider using a smaller model or optimizing prompts",
                "expected_improvement": "30-50% faster responses"
            })
        
        # Success rate recommendations
        if analysis["success_rate"] < 0.9:
            recommendations.append({
                "type": "reliability",
                "priority": "high",
                "issue": "Low success rate",
                "recommendation": "Review error patterns and improve error handling",
                "expected_improvement": "Improved reliability"
            })
        
        # Variability recommendations
        if analysis["response_time_std"] > analysis["avg_response_time"] * 0.5:
            recommendations.append({
                "type": "consistency",
                "priority": "medium",
                "issue": "High response time variability",
                "recommendation": "Implement response caching for common queries",
                "expected_improvement": "More consistent performance"
            })
        
        return recommendations
    
    async def optimize_model_settings(self, model_name: str) -> Dict[str, Any]:
        """Automatically optimize model settings."""
        # Placeholder for automatic optimization
        # This would involve testing different parameters and measuring performance
        
        optimization_result = {
            "model_name": model_name,
            "optimization_timestamp": datetime.now().isoformat(),
            "status": "completed",
            "improvements": [],
            "new_settings": {}
        }
        
        self.optimization_history.append(optimization_result)
        return optimization_result
