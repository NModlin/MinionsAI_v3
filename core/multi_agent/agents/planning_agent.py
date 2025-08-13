"""
MinionsAI v3.1 - Planning Agent
Specialized agent for task decomposition, workflow planning, and coordination.
"""

from typing import Dict, List, Any, Optional
import logging
from datetime import datetime, timedelta

from .base_agent import BaseAgent
from ..agent_registry import AgentCapability

logger = logging.getLogger(__name__)


class PlanningAgent(BaseAgent):
    """
    Specialized agent for planning and coordination tasks.
    """
    
    def __init__(self, model_name: str = "llama3:8b", base_url: str = "http://localhost:11434"):
        """Initialize the Planning Agent."""
        super().__init__(
            name="Planning Agent",
            description="Specialized in task decomposition, workflow planning, and coordination",
            capabilities=[
                AgentCapability.TASK_PLANNING,
                AgentCapability.NATURAL_LANGUAGE,
                AgentCapability.INFORMATION_SYNTHESIS
            ],
            model_name=model_name,
            base_url=base_url
        )
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for the Planning Agent."""
        return """You are a Planning Agent, specialized in breaking down complex tasks, creating workflows, and coordinating multi-step processes.

Your capabilities include:
- Decomposing complex tasks into manageable subtasks
- Creating detailed project plans and timelines
- Identifying task dependencies and critical paths
- Resource allocation and scheduling
- Risk assessment and mitigation planning
- Progress tracking and milestone definition
- Workflow optimization and process improvement

When creating plans:
1. Analyze the overall objective thoroughly
2. Break down into logical, sequential steps
3. Identify dependencies between tasks
4. Estimate time and resource requirements
5. Consider potential risks and alternatives
6. Create clear milestones and success criteria
7. Ensure plans are actionable and measurable

Always provide structured, detailed plans that can be easily followed and executed."""
    
    async def execute_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a planning task."""
        try:
            task_type = task_data.get("task_type", "create_plan")
            
            logger.info(f"Planning Agent executing {task_type}")
            
            if task_type == "create_plan":
                return await self._create_plan(task_data)
            elif task_type == "decompose_task":
                return await self._decompose_task(task_data)
            elif task_type == "create_workflow":
                return await self._create_workflow(task_data)
            elif task_type == "optimize_plan":
                return await self._optimize_plan(task_data)
            elif task_type == "assess_risks":
                return await self._assess_risks(task_data)
            else:
                return await self._general_planning(task_data)
                
        except Exception as e:
            logger.error(f"Error in Planning Agent task execution: {e}")
            return {
                "success": False,
                "error": str(e),
                "plan": {}
            }
    
    async def _create_plan(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a comprehensive plan for a given objective."""
        objective = task_data.get("objective", "")
        constraints = task_data.get("constraints", [])
        timeline = task_data.get("timeline", "")
        resources = task_data.get("resources", [])
        
        if not objective:
            return {
                "success": False,
                "error": "No objective provided for planning"
            }
        
        planning_prompt = f"""
        Create a comprehensive plan for the following objective:
        
        Objective: {objective}
        Timeline: {timeline}
        Constraints: {', '.join(constraints) if constraints else 'None specified'}
        Available Resources: {', '.join(resources) if resources else 'None specified'}
        
        Please provide a detailed plan including:
        1. Executive Summary
        2. Task Breakdown Structure
        3. Timeline and Milestones
        4. Resource Requirements
        5. Risk Assessment
        6. Success Criteria
        7. Dependencies and Critical Path
        8. Contingency Plans
        
        Format the plan clearly with numbered sections and actionable items.
        """
        
        plan_response = await self.generate_response(planning_prompt)
        
        # Extract structured plan components
        plan_structure = self._parse_plan_structure(plan_response)
        
        return {
            "success": True,
            "task_type": "create_plan",
            "objective": objective,
            "plan": plan_response,
            "plan_structure": plan_structure,
            "constraints": constraints,
            "timeline": timeline,
            "resources": resources,
            "creation_timestamp": datetime.now().isoformat()
        }
    
    async def _decompose_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Decompose a complex task into subtasks."""
        main_task = task_data.get("task", "")
        complexity_level = task_data.get("complexity", "medium")
        
        if not main_task:
            return {
                "success": False,
                "error": "No task provided for decomposition"
            }
        
        decomposition_prompt = f"""
        Decompose the following complex task into manageable subtasks:
        
        Main Task: {main_task}
        Complexity Level: {complexity_level}
        
        Please provide:
        1. List of subtasks in logical order
        2. Dependencies between subtasks
        3. Estimated effort for each subtask
        4. Required skills/capabilities for each subtask
        5. Priority levels
        6. Potential parallel execution opportunities
        
        Format as a structured breakdown with clear relationships.
        """
        
        decomposition = await self.generate_response(decomposition_prompt)
        
        # Parse subtasks
        subtasks = self._parse_subtasks(decomposition)
        
        return {
            "success": True,
            "task_type": "decompose_task",
            "main_task": main_task,
            "decomposition": decomposition,
            "subtasks": subtasks,
            "complexity_level": complexity_level,
            "decomposition_timestamp": datetime.now().isoformat()
        }
    
    async def _create_workflow(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a workflow for a process."""
        process_description = task_data.get("process", "")
        workflow_type = task_data.get("type", "sequential")
        
        if not process_description:
            return {
                "success": False,
                "error": "No process description provided"
            }
        
        workflow_prompt = f"""
        Create a detailed workflow for the following process:
        
        Process: {process_description}
        Workflow Type: {workflow_type}
        
        Please provide:
        1. Workflow steps in sequence
        2. Decision points and branching logic
        3. Input/output for each step
        4. Responsible roles or agents
        5. Quality checkpoints
        6. Error handling procedures
        7. Performance metrics
        
        Create a clear, actionable workflow that can be implemented.
        """
        
        workflow = await self.generate_response(workflow_prompt)
        
        # Parse workflow steps
        workflow_steps = self._parse_workflow_steps(workflow)
        
        return {
            "success": True,
            "task_type": "create_workflow",
            "process": process_description,
            "workflow": workflow,
            "workflow_steps": workflow_steps,
            "workflow_type": workflow_type,
            "creation_timestamp": datetime.now().isoformat()
        }
    
    async def _optimize_plan(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize an existing plan."""
        current_plan = task_data.get("plan", "")
        optimization_goals = task_data.get("goals", ["efficiency", "cost", "time"])
        
        if not current_plan:
            return {
                "success": False,
                "error": "No plan provided for optimization"
            }
        
        optimization_prompt = f"""
        Optimize the following plan based on these goals: {', '.join(optimization_goals)}
        
        Current Plan:
        {current_plan}
        
        Please provide:
        1. Analysis of current plan inefficiencies
        2. Optimized version of the plan
        3. Specific improvements made
        4. Expected benefits of optimization
        5. Trade-offs and considerations
        6. Implementation recommendations
        
        Focus on achieving the optimization goals while maintaining plan effectiveness.
        """
        
        optimization = await self.generate_response(optimization_prompt)
        
        return {
            "success": True,
            "task_type": "optimize_plan",
            "original_plan": current_plan,
            "optimization": optimization,
            "optimization_goals": optimization_goals,
            "optimization_timestamp": datetime.now().isoformat()
        }
    
    async def _assess_risks(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risks for a plan or project."""
        plan_or_project = task_data.get("plan", "")
        risk_categories = task_data.get("categories", ["technical", "schedule", "resource", "external"])
        
        if not plan_or_project:
            return {
                "success": False,
                "error": "No plan or project provided for risk assessment"
            }
        
        risk_prompt = f"""
        Perform a comprehensive risk assessment for the following plan/project:
        
        Plan/Project:
        {plan_or_project}
        
        Risk Categories to Assess: {', '.join(risk_categories)}
        
        Please provide:
        1. Identified risks by category
        2. Risk probability and impact assessment
        3. Risk priority matrix
        4. Mitigation strategies for high-priority risks
        5. Contingency plans
        6. Risk monitoring recommendations
        7. Overall risk assessment summary
        
        Use a structured approach to risk identification and assessment.
        """
        
        risk_assessment = await self.generate_response(risk_prompt)
        
        # Parse risk information
        risks = self._parse_risks(risk_assessment)
        
        return {
            "success": True,
            "task_type": "assess_risks",
            "plan_or_project": plan_or_project,
            "risk_assessment": risk_assessment,
            "identified_risks": risks,
            "risk_categories": risk_categories,
            "assessment_timestamp": datetime.now().isoformat()
        }
    
    async def _general_planning(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle general planning requests."""
        description = task_data.get("description", "")
        
        if not description:
            return {
                "success": False,
                "error": "No planning description provided"
            }
        
        general_prompt = f"""
        Address the following planning request:
        
        Request: {description}
        
        Provide appropriate planning guidance, structure, or recommendations based on the request.
        """
        
        response = await self.generate_response(general_prompt)
        
        return {
            "success": True,
            "task_type": "general_planning",
            "description": description,
            "response": response,
            "timestamp": datetime.now().isoformat()
        }
    
    # Helper methods
    def _parse_plan_structure(self, plan_text: str) -> Dict[str, Any]:
        """Parse plan text to extract structured components."""
        # Simple parsing - could be enhanced with more sophisticated NLP
        structure = {
            "sections": [],
            "tasks": [],
            "milestones": [],
            "risks": []
        }
        
        lines = plan_text.split('\n')
        current_section = ""
        
        for line in lines:
            line = line.strip()
            if line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.')):
                current_section = line
                structure["sections"].append(current_section)
            elif line.startswith(('-', '*', '•')):
                if 'task' in current_section.lower() or 'action' in current_section.lower():
                    structure["tasks"].append(line[1:].strip())
                elif 'milestone' in current_section.lower():
                    structure["milestones"].append(line[1:].strip())
                elif 'risk' in current_section.lower():
                    structure["risks"].append(line[1:].strip())
        
        return structure
    
    def _parse_subtasks(self, decomposition_text: str) -> List[Dict[str, Any]]:
        """Parse decomposition text to extract subtasks."""
        subtasks = []
        lines = decomposition_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith(('-', '*', '•')) or line[0:2].isdigit():
                subtask = {
                    "description": line.lstrip('-*•0123456789. '),
                    "priority": "normal",
                    "estimated_effort": "medium",
                    "dependencies": []
                }
                subtasks.append(subtask)
        
        return subtasks
    
    def _parse_workflow_steps(self, workflow_text: str) -> List[Dict[str, Any]]:
        """Parse workflow text to extract steps."""
        steps = []
        lines = workflow_text.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith(('-', '*', '•')) or (line and line[0].isdigit()):
                step = {
                    "step_number": len(steps) + 1,
                    "description": line.lstrip('-*•0123456789. '),
                    "type": "action",
                    "dependencies": []
                }
                steps.append(step)
        
        return steps
    
    def _parse_risks(self, risk_text: str) -> List[Dict[str, Any]]:
        """Parse risk assessment text to extract risks."""
        risks = []
        lines = risk_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith(('-', '*', '•')) and ('risk' in line.lower() or 'probability' in line.lower()):
                risk = {
                    "description": line.lstrip('-*• '),
                    "category": "general",
                    "probability": "medium",
                    "impact": "medium",
                    "priority": "medium"
                }
                risks.append(risk)
        
        return risks
    
    async def _initialize_agent(self) -> None:
        """Planning Agent specific initialization."""
        self.add_to_memory("planning_history", [])
        self.add_to_memory("workflow_templates", {})
        logger.info("Planning Agent initialized with planning capabilities")
    
    async def _cleanup_agent(self) -> None:
        """Planning Agent specific cleanup."""
        logger.info("Planning Agent cleanup completed")
