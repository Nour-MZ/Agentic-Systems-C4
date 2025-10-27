"""
ReAct Agent with Personas: Fitness & Wellness Coach
Framework: LangGraph with Direct Google AI API
Course: C4 Assignment - 503P/798S

This version uses Google's generativeai library directly (most reliable)

Installation:
pip install langgraph google-generativeai python-dotenv

Setup:
export GOOGLE_API_KEY="your-free-key"
Get key at: https://makersuite.google.com/app/apikey
"""

import os
import re
import json
from typing import TypedDict, Literal
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
import google.generativeai as genai

load_dotenv()

# Configure Google AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ============================================================================
# STATE
# ============================================================================

class AgentState(TypedDict):
    messages: list
    iterations: int
    max_iterations: int

# ============================================================================
# TOOLS
# ============================================================================

def lookup_exercise_info(exercise_name: str) -> str:
    """Get exercise information"""
    db = {
        "squats": "Squats: Lower body compound. Targets quads, glutes, hamstrings. Beginner: 3x10 bodyweight. Form: chest up, knees over toes.",
        "pushups": "Push-ups: Upper body compound. Targets chest, shoulders, triceps. Beginner: 3x8 knee pushups. Form: straight body line.",
        "deadlifts": "Deadlifts: Full body compound. Targets back, glutes. REQUIRES proper form. Start 3x5 light. Form: neutral spine, hinge at hips.",
        "running": "Running: Cardio. Improves cardiovascular health. Beginner: 20 min, 3x/week. Progress gradually.",
        "plank": "Plank: Core stability. Beginner: 3x30sec. Advanced: 3x2min. Form: straight line head to heels.",
        "yoga": "Yoga: Flexibility and mindfulness. Start with 20-min beginner sessions. Reduces stress.",
        "bench press": "Bench Press: Upper body push. Chest, shoulders, triceps. Beginner: 3x8 light weight.",
        "pull-ups": "Pull-ups: Upper body pull. Back, biceps. Very challenging. Start with assisted 3x5.",
        "lunges": "Lunges: Unilateral lower body. Quads, glutes, balance. Beginner: 3x10 each leg bodyweight.",
        "burpees": "Burpees: Full body cardio. High calorie burn. Beginner: 3x5. Full movement: squat-plank-jump."
    }
    
    for key in db:
        if key in exercise_name.lower():
            return db[key]
    return f"No info for '{exercise_name}'. Try: squats, pushups, deadlifts, running, plank, yoga, bench press, pull-ups, lunges, burpees."

def calculate_bmr(weight_kg: float, height_cm: float, age: int, gender: str) -> str:
    """Calculate BMR"""
    try:
        weight_kg, height_cm, age = float(weight_kg), float(height_cm), int(age)
        
        if gender.lower() == "male":
            bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
        else:
            bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161
        
        return f"BMR: {bmr:.0f} cal/day. Activity multipliers: Sedentary(1.2), Light(1.375), Moderate(1.55), Active(1.725), Very Active(1.9)"
    except:
        return "Error: Invalid parameters. Need: weight_kg(number), height_cm(number), age(number), gender(male/female)"

def get_workout_plan(fitness_level: str, goal: str, days_per_week: str) -> str:
    """Generate workout plan"""
    plans = {
        "beginner_weight_loss_3": """BEGINNER WEIGHT LOSS (3 days/week):
• Monday: 30min brisk walk + Bodyweight circuit (squats 3x12, pushups 3x8, plank 3x30s)
• Wednesday: 25min light jog + Core work (crunches 3x15, leg raises 3x10)
• Friday: Yoga 30min + Bodyweight (lunges 3x10, wall sits 3x30s)
Rest: Active recovery on other days""",
        
        "beginner_muscle_gain_3": """BEGINNER MUSCLE GAIN (3 days/week):
• Monday: Upper Body (pushups 3x8, dumbbell rows 3x10, shoulder press 3x10, bicep curls 3x12)
• Wednesday: Lower Body (squats 3x12, lunges 3x10 each, Romanian deadlifts 3x10, calf raises 3x15)
• Friday: Full Body Circuit (all movements, lighter weight, 3 rounds)
Progressive overload: Add 1-2 reps weekly""",
        
        "intermediate_weight_loss_4": """INTERMEDIATE WEIGHT LOSS (4 days/week):
• Monday: HIIT cardio 25min (30s sprint, 30s rest)
• Tuesday: Upper strength (bench 3x10, rows 3x10, shoulder work)
• Thursday: Lower strength (squats 3x12, deadlifts 3x8, leg press 3x12)
• Saturday: Steady cardio 45min (run, bike, swim)
Nutrition: 300-500 cal deficit""",
        
        "intermediate_muscle_gain_4": """INTERMEDIATE MUSCLE GAIN (4 days/week):
• Monday: Chest & Triceps (bench press 4x8, incline 3x10, dips 3x12, tricep extensions 3x12)
• Tuesday: Back & Biceps (deadlifts 4x6, pull-ups 3x8, rows 3x10, curls 3x12)
• Thursday: Legs (squats 4x8, leg press 3x12, hamstring curls 3x12, calf raises 4x15)
• Friday: Shoulders & Core (overhead press 4x8, lateral raises 3x12, planks 3x60s)
Progressive overload: Increase weight at top rep range""",
        
        "advanced_muscle_gain_5": """ADVANCED MUSCLE GAIN (5 days/week - PPL):
• Monday: Chest (bench 4x6, incline 4x8, flies 3x12, close-grip 3x10)
• Tuesday: Back (deadlifts 5x5, weighted pull-ups 4x6, rows 4x8, face pulls 3x15)
• Wednesday: Legs (squats 5x5, front squats 3x8, RDLs 3x10, leg extensions 3x15, ham curls 3x12)
• Thursday: Shoulders (OHP 4x6, Arnold press 3x10, lateral raises 4x12, rear delts 3x15)
• Friday: Arms (barbell curls 4x8, hammer curls 3x12, dips 4x8, overhead extensions 3x12)
Periodization: Vary rep ranges every 4 weeks"""
    }
    
    key = f"{fitness_level.lower()}_{goal.lower().replace(' ', '_')}_{days_per_week}"
    return plans.get(key, f"{fitness_level.upper()} {goal} plan: {days_per_week} days/week. Mix strength training + cardio + rest days.")

def nutrition_advice(goal: str, dietary_restrictions: str = "none") -> str:
    """Nutrition guidance"""
    advice = {
        "weight_loss": """WEIGHT LOSS NUTRITION:
Caloric Target: 300-500 cal deficit from maintenance
• Protein: 1.8-2.2g/kg (preserve muscle, increase satiety)
• Fats: 0.8-1g/kg (hormones)
• Carbs: Fill remaining (prioritize around workouts)
Key: High protein meals, lots of vegetables, limit processed foods, stay hydrated""",
        
        "muscle_gain": """MUSCLE GAIN NUTRITION:
Caloric Target: 200-400 cal surplus
• Protein: 2.0-2.4g/kg (muscle synthesis)
• Carbs: 4-6g/kg (fuel + recovery)
• Fats: 0.8-1.2g/kg
Key: Protein every 3-4 hours, carbs pre/post workout, slight surplus, whole foods priority""",
        
        "maintenance": """MAINTENANCE NUTRITION:
Caloric Target: At TDEE (BMR × activity level)
• Protein: 1.6-2.0g/kg
• Carbs: 3-5g/kg
• Fats: 0.8-1.2g/kg
Key: Balanced macros, 80/20 rule (80% whole foods), listen to hunger, food quality"""
    }
    
    base = advice.get(goal.lower().replace(" ", "_"), advice["maintenance"])
    
    if "vegetarian" in dietary_restrictions.lower():
        base += "\n\nVEGETARIAN: Protein from eggs, dairy, legumes, tofu, tempeh. Ensure B12."
    elif "vegan" in dietary_restrictions.lower():
        base += "\n\nVEGAN: Protein from legumes, tofu, tempeh, seitan, nuts. Supplement B12, D3, Omega-3."
    elif "gluten" in dietary_restrictions.lower():
        base += "\n\nGLUTEN-FREE: Use rice, quinoa, oats (certified GF), buckwheat. Check labels."
    
    return base

TOOLS = {
    "lookup_exercise_info": lookup_exercise_info,
    "calculate_bmr": calculate_bmr,
    "get_workout_plan": get_workout_plan,
    "nutrition_advice": nutrition_advice
}

# ============================================================================
# REACT PARSER
# ============================================================================

def parse_react(text: str) -> dict:
    """Parse ReAct format"""
    result = {"thought": "", "action": "", "action_input": {}, "final_answer": ""}
    
    # Thought
    thought = re.search(r'Thought:\s*(.+?)(?=\n(?:Action:|Final Answer:)|$)', text, re.DOTALL | re.IGNORECASE)
    if thought:
        result["thought"] = thought.group(1).strip()
    
    # Action
    action = re.search(r'Action:\s*(\w+)', text, re.IGNORECASE)
    if action:
        result["action"] = action.group(1).strip()
    
    # Action Input
    action_input = re.search(r'Action Input:\s*(.+?)(?=\nObservation:|$)', text, re.DOTALL | re.IGNORECASE)
    if action_input:
        input_text = action_input.group(1).strip()
        try:
            result["action_input"] = json.loads(input_text)
        except:
            # Extract quoted strings and numbers
            params = re.findall(r'["\']([^"\']+)["\']|(\d+\.?\d*)', input_text)
            result["action_input"] = [p[0] if p[0] else p[1] for p in params]
    
    # Final Answer
    answer = re.search(r'Final Answer:\s*(.+)', text, re.DOTALL | re.IGNORECASE)
    if answer:
        result["final_answer"] = answer.group(1).strip()
    
    return result

def execute_tool(tool_name: str, tool_input) -> str:
    """Execute tool"""
    if tool_name not in TOOLS:
        return f"Unknown tool: {tool_name}"
    
    try:
        tool_func = TOOLS[tool_name]
        if isinstance(tool_input, dict):
            return tool_func(**tool_input)
        elif isinstance(tool_input, list):
            return tool_func(*tool_input)
        else:
            return tool_func(tool_input)
    except Exception as e:
        return f"Error executing {tool_name}: {str(e)}"

# ============================================================================
# PERSONAS
# ============================================================================

PERSONAS = {
    "friendly_advisor": {
        "name": "Friendly Advisor",
        "system_prompt": """You are Alex, a friendly and encouraging fitness coach.

TOOLS AVAILABLE:
- lookup_exercise_info(exercise_name)
- calculate_bmr(weight_kg, height_cm, age, gender)
- get_workout_plan(fitness_level, goal, days_per_week)
- nutrition_advice(goal, dietary_restrictions)

REACT FORMAT - USE EXACTLY:
Thought: [What information do I need?]
Action: tool_name
Action Input: "param1", "param2", "param3"
Observation: [Result will appear here]
... (repeat Thought/Action/Observation as needed)
Thought: I now have enough information
Final Answer: [Your warm, encouraging, complete response]

EXAMPLE:
Thought: User wants beginner weight loss plan. I need workout plan and nutrition advice.
Action: get_workout_plan
Action Input: "beginner", "weight_loss", "3"
Observation: [workout details]
Thought: Now get nutrition advice
Action: nutrition_advice
Action Input: "weight_loss", "none"
Observation: [nutrition details]
Thought: I have everything needed
Final Answer: Awesome goal! Here's your personalized plan...

Be supportive and use tools for accurate info!""",
        "temperature": 0.7
    },
    
    "strict_expert": {
        "name": "Strict Expert",
        "system_prompt": """You are Dr. Stevens, an exercise physiologist. Direct and evidence-based.

TOOLS AVAILABLE:
- lookup_exercise_info(exercise_name)
- calculate_bmr(weight_kg, height_cm, age, gender)
- get_workout_plan(fitness_level, goal, days_per_week)
- nutrition_advice(goal, dietary_restrictions)

REACT FORMAT - MANDATORY:
Thought: [Analytical assessment]
Action: tool_name
Action Input: "param1", "param2", "param3"
Observation: [Tool output]
... (iterate until data complete)
Thought: Sufficient data collected
Final Answer: [Precise technical recommendation with metrics]

Be professional, factual, use specific numbers.""",
        "temperature": 0.3
    },
    
    "cautious_helper": {
        "name": "Cautious Helper",
        "system_prompt": """You are Coach Maria, specializing in injury prevention. Safety first!

TOOLS AVAILABLE:
- lookup_exercise_info(exercise_name)
- calculate_bmr(weight_kg, height_cm, age, gender)
- get_workout_plan(fitness_level, goal, days_per_week)
- nutrition_advice(goal, dietary_restrictions)

REACT FORMAT - REQUIRED:
Thought: [Consider safety]
Action: tool_name
Action Input: "param1", "param2", "param3"
Observation: [Result]
... (gather safety info)
Thought: Can provide safe recommendation
Final Answer: [Safety-conscious guidance with warnings and modifications]

Always mention form, provide easier alternatives, start conservatively.""",
        "temperature": 0.5
    }
}

# ============================================================================
# LANGGRAPH NODES
# ============================================================================

def should_continue(state: AgentState) -> Literal["tools", "agent", "end"]:
    """Routing logic"""
    if state["iterations"] >= state["max_iterations"]:
        return "end"
    
    last_msg = state["messages"][-1]
    content = last_msg["content"].lower()
    
    if "final answer:" in content:
        return "end"
    
    if "action:" in content and "action input:" in content:
        return "tools"
    
    return "agent"

def agent_node(state: AgentState, model, system_prompt: str) -> dict:
    """Call Gemini properly using combined text prompt"""
    # Combine system prompt + conversation history into a single text block
    conversation_text = system_prompt + "\n\n"
    for msg in state["messages"]:
        role = msg.get("role", "user").upper()
        content = msg.get("content", "")
        conversation_text += f"{role}: {content}\n"

    # Generate model response
    response = model.generate_content(conversation_text)

    return {
        "messages": [
            {
                "role": "model",
                "parts": [response.text],
                "content": response.text
            }
        ],
        "iterations": state["iterations"] + 1
    }

def tools_node(state: AgentState) -> dict:
    """Execute tools"""
    last_msg = state["messages"][-1]
    parsed = parse_react(last_msg["content"])
    
    if parsed["action"]:
        observation = execute_tool(parsed["action"], parsed["action_input"])
        return {
            "messages": [{
                "role": "model",
                "parts": [f"Observation: {observation}\n\n"],
                "content": f"Observation: {observation}\n\n"
            }]
        }
    
    return {}

# ============================================================================
# REACT AGENT
# ============================================================================

class ReActFitnessAgent:
    """LangGraph ReAct agent with direct Google AI"""
    
    def __init__(self, persona_key: str):
        self.persona = PERSONAS[persona_key]
        
        # Initialize Gemini model directly
        self.model = genai.GenerativeModel(
            'gemini-2.0-flash-exp',
            generation_config=genai.types.GenerationConfig(
                temperature=self.persona["temperature"],
                max_output_tokens=2048,
            )
        )
        
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Build LangGraph"""
        workflow = StateGraph(AgentState)
        
        # Nodes
        workflow.add_node("agent", lambda s: agent_node(s, self.model, self.persona["system_prompt"]))
        workflow.add_node("tools", tools_node)
        
        # Entry
        workflow.set_entry_point("agent")
        
        # Edges
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {"tools": "tools", "agent": "agent", "end": END}
        )
        workflow.add_edge("tools", "agent")
        
        return workflow.compile()
    
    def run(self, query: str, max_iterations: int = 6, verbose: bool = True) -> dict:
        """Run ReAct loop"""
        initial_state = {
            "messages": [{"role": "user", "parts": [query], "content": query}],
            "iterations": 0,
            "max_iterations": max_iterations
        }
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"PERSONA: {self.persona['name']}")
            print(f"TEMP: {self.persona['temperature']}")
            print(f"QUERY: {query}")
            print(f"{'='*80}\n")
        
        # Run
        final_state = self.graph.invoke(initial_state)
        
        # Extract response
        final_msg = final_state["messages"][-1]["content"]
        parsed = parse_react(final_msg)
        response = parsed["final_answer"] if parsed["final_answer"] else final_msg
        
        if verbose:
            print(f"\nTRAJECTORY ({final_state['iterations']} iterations):")
            print("="*80)
            for i, msg in enumerate(final_state["messages"]):
                print(f"\n[Step {i+1}]")
                content = msg["content"]
                print(content[:400] + "..." if len(content) > 400 else content)
            
            print(f"\n{'='*80}")
            print(f"FINAL ANSWER:")
            print(f"{'='*80}")
            print(response)
        
        return {
            "persona": self.persona["name"],
            "query": query,
            "response": response,
            "iterations": final_state["iterations"]
        }

# ============================================================================
# TESTING
# ============================================================================

def test_single(query: str, persona: str):
    """Test one persona"""
    agent = ReActFitnessAgent(persona)
    return agent.run(query)

def test_all_personas(query: str):
    """Compare all personas"""
    print(f"\n{'#'*80}")
    print(f"# PERSONA COMPARISON")
    print(f"# {query}")
    print(f"{'#'*80}\n")
    
    results = {}
    for p in PERSONAS.keys():
        results[p] = test_single(query, p)
        print("\n")
    return results

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    if not os.getenv("GOOGLE_API_KEY"):
        print("ERROR: Set GOOGLE_API_KEY")
        print("Get free key: https://makersuite.google.com/app/apikey")
        exit(1)
    
    print("="*80)
    print(" ReAct Fitness Agent - LangGraph + Google AI")
    print(" C4 Assignment - 503P/798S")
    print("="*80)
    
    # Example
    # test_single(
    #     "I want to build muscle. Give me a 4-day workout plan.",
    #     "friendly_advisor"
    # )
    
    # Uncomment to test all:
    test_all_personas("Tell me about deadlifts and if they're safe for beginners")