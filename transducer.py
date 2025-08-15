from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Union, Optional
import re
from collections import deque, defaultdict
from enum import Enum

# ---------------------------
# Graph-based STL Transducer Components
# ---------------------------

class TransducerState(Enum):
    """States for transducer automata"""
    INACTIVE = 0
    ACTIVE = 1
    SUCCESS = 2
    FAILURE = 3

def parse_formula(formula: str):
    """Parse STL formula into abstract syntax tree"""
    formula = formula.replace(" ", "")
    # Improved tokenization to handle predicates and operators separately
    tokens = re.findall(r'[UFGX]\[\d+,\d+\]|\w+[><=]+\w+|\w+[><=]+[\d.]+|[!&|UFGX()]|\w+', formula)
    
    def parse_tokens(tokens):
        tokens = deque(tokens)
        
        def parse_primary():
            if not tokens:
                raise ValueError("Unexpected end of formula")
            
            token = tokens.popleft()
            
            if token == '(':
                node = parse_expression()
                if not tokens or tokens.popleft() != ')':
                    raise ValueError("Missing closing parenthesis")
                return node
            elif token.startswith('F['):
                # Extract interval from F[a,b]
                match = re.match(r'F\[(\d+),(\d+)\]', token)
                if not match:
                    raise ValueError(f"Invalid F operator format: {token}")
                a, b = int(match.group(1)), int(match.group(2))
                if not tokens or tokens.popleft() != '(':
                    raise ValueError("Expected '(' after F[a,b]")
                arg = parse_expression()
                if not tokens or tokens.popleft() != ')':
                    raise ValueError("Missing closing parenthesis after F operator")
                return {'op': 'F', 'interval': (a, b), 'arg': arg}
            elif token.startswith('G['):
                # Extract interval from G[a,b]
                match = re.match(r'G\[(\d+),(\d+)\]', token)
                if not match:
                    raise ValueError(f"Invalid G operator format: {token}")
                a, b = int(match.group(1)), int(match.group(2))
                if not tokens or tokens.popleft() != '(':
                    raise ValueError("Expected '(' after G[a,b]")
                inner_arg = parse_expression()
                if not tokens or tokens.popleft() != ')':
                    raise ValueError("Missing closing parenthesis after G operator")
                return {'op': 'G', 'interval': (a, b), 'arg': inner_arg}
            elif token == 'X':
                if not tokens or tokens.popleft() != '(':
                    raise ValueError("Expected '(' after X")
                arg = parse_expression()
                if not tokens or tokens.popleft() != ')':
                    raise ValueError("Missing closing parenthesis after X operator")
                return {
                    'op': 'U',
                    'interval': (1, 1),
                    'args': ['false', arg]
                }
            elif token == '!':
                arg = parse_primary()
                return {'op': '!', 'arg': arg}
            else:
                # This should be an atomic predicate or variable
                return token

        def parse_expression():
            lhs = parse_primary()
            while tokens and tokens[0] in ('&', '|', 'U'):
                op = tokens.popleft()
                
                # Handle U[a,b] format
                if op == 'U' and tokens and tokens[0].startswith('['):
                    interval_token = tokens.popleft()
                    match = re.match(r'\[(\d+),(\d+)\]', interval_token)
                    if not match:
                        raise ValueError(f"Invalid interval format: {interval_token}")
                    a, b = int(match.group(1)), int(match.group(2))
                    rhs = parse_primary()
                    lhs = {'op': op, 'args': [lhs, rhs], 'interval': (a, b)}
                else:
                    rhs = parse_primary()
                    expr = {'op': op, 'args': [lhs, rhs]}
                    if op == 'U':
                        expr['interval'] = (0, float('inf'))  # Default unbounded until
                    lhs = expr
            return lhs

        return parse_expression()

    return parse_tokens(tokens)

class Transducer(ABC):
    """Base class for STL transducers with graph-based state management"""
    
    def __init__(self, interval: Tuple[int, int] = (0, 0), parent: Optional['Transducer'] = None):
        self.interval = interval
        self.parent = parent
        self.children = []
        self.t = 0  # current time
        self.state = TransducerState.INACTIVE
        self.state_history = []  # Track state changes over time
        self.output_history = []  # Track outputs over time
        self.satisfied = False  # Track permanent satisfaction
        self.violated = False   # Track permanent violation
        
    def reset(self):
        """Reset transducer to initial state"""
        self.t = 0
        self.state = TransducerState.INACTIVE
        self.state_history.clear()
        self.output_history.clear()
        self.satisfied = False
        self.violated = False
        for child in self.children:
            child.reset()
    
    def step(self, signal: Union[Tuple[float, float], List[float]], time: float) -> Optional[bool]:
        """Process one time step and return output (supports continuous time). Signal is (x, y) or [x, y]."""
        self.t = time
        
        # If permanently satisfied or violated, return that state
        if self.satisfied:
            return True
        if self.violated:
            return False
            
        output = self.trans(signal, time)
        self.state_history.append(self.state)
        self.output_history.append(output)
        return output
    
    @abstractmethod
    def trans(self, signal: Union[Tuple[float, float], List[float]], time: float) -> Optional[bool]:
        """Transition function - must be implemented by subclasses. Signal is (x, y) or [x, y]."""
        pass

class AtomicTransducer(Transducer):
    """Atomic predicate transducer"""
    def __init__(self, predicate: str, parent: Optional[Transducer] = None):
        super().__init__(parent=parent)
        self.predicate = predicate
        self.center = None  # Center for circle region predicates
        self.radius = None  # Radius for circle region predicates
        
    def set_region(self, center: list, radius: float):
        """Set the center and radius for circle region predicates"""
        self.center = center
        self.radius = radius
    
    def trans(self, signal_point: Tuple[float, float], time: float) -> Optional[bool]:
        """Evaluate atomic predicate for a single 2D point (x, y)"""
        if self.center is None or self.radius is None:
            raise ValueError(f"Region not set for predicate {self.predicate}")
        
        x, y = signal_point
        dist = ((x - self.center[0]) ** 2 + (y - self.center[1]) ** 2) ** 0.5
        result = dist <= self.radius
        
        self.state = TransducerState.SUCCESS if result else TransducerState.FAILURE
        return True if result else False
       
class NotTransducer(Transducer):
    """Negation transducer"""
    
    def __init__(self, child: Transducer, parent: Optional[Transducer] = None):
        super().__init__(parent=parent)
        self.child = child
        self.children = [child]
        child.parent = self
    
    def trans(self, signal: Union[Tuple[float, float], List[float]], time: float) -> Optional[bool]:
        """Negate child output with proper None handling"""
        child_output = self.child.step(signal, time)
        
        # Handle permanent states first
        if self.child.satisfied:
            self.violated = True
            return False
        if self.child.violated:
            self.satisfied = True
            return True
            
        # Standard negation logic: !None = None, !True = False, !False = True
        if child_output is None:
            self.state = TransducerState.INACTIVE
            return None
        
        result = not child_output
        self.state = TransducerState.SUCCESS if result else TransducerState.FAILURE
        return result

class AndTransducer(Transducer):
    """Conjunction transducer with proper RRT logic"""
    
    def __init__(self, left: Transducer, right: Transducer, parent: Optional[Transducer] = None):
        super().__init__(parent=parent)
        self.left = left
        self.right = right
        self.children = [left, right]
        left.parent = self
        right.parent = self
    
    def trans(self, signal: Union[Tuple[float, float], List[float]], time: float) -> Optional[bool]:
        """Conjunction logic: None & None = None, False & X = False, True & None = None, True & True = True"""
        left_output = self.left.step(signal, time)
        right_output = self.right.step(signal, time)
        
        # Handle permanent violations - if either child is violated, AND is violated
        if self.left.violated or self.right.violated:
            self.violated = True
            return False
            
        # Handle permanent satisfaction - both must be satisfied
        if self.left.satisfied and self.right.satisfied:
            self.satisfied = True
            return True
        
        # Standard AND logic with None handling
        if left_output is False or right_output is False:
            self.state = TransducerState.FAILURE
            return False
        elif left_output is True and right_output is True:
            self.state = TransducerState.SUCCESS
            return True
        elif left_output is None or right_output is None:
            # None & None = None, True & None = None, None & True = None
            self.state = TransducerState.INACTIVE
            return None
        else:
            # Should not reach here, but default to None for safety
            self.state = TransducerState.ACTIVE
            return None

class OrTransducer(Transducer):
    """Disjunction transducer with proper RRT logic"""
    
    def __init__(self, left: Transducer, right: Transducer, parent: Optional[Transducer] = None):
        super().__init__(parent=parent)
        self.left = left
        self.right = right
        self.children = [left, right]
        left.parent = self
        right.parent = self
    
    def trans(self, signal: Union[Tuple[float, float], List[float]], time: float) -> Optional[bool]:
        """Disjunction logic: None | None = None, True | X = True, False | None = None, False | False = False"""
        left_output = self.left.step(signal, time)
        right_output = self.right.step(signal, time)
        
        # Handle permanent satisfaction - if either child is satisfied, OR is satisfied
        if self.left.satisfied or self.right.satisfied:
            self.satisfied = True
            return True
            
        # Handle permanent violation - both must be violated
        if self.left.violated and self.right.violated:
            self.violated = True
            return False
        
        # Standard OR logic with None handling
        if left_output is True or right_output is True:
            self.state = TransducerState.SUCCESS
            return True
        elif left_output is False and right_output is False:
            self.state = TransducerState.FAILURE
            return False
        elif left_output is None or right_output is None:
            # None | None = None, False | None = None, None | False = None
            self.state = TransducerState.INACTIVE
            return None
        else:
            # Should not reach here, but default to None for safety
            self.state = TransducerState.ACTIVE
            return None

class EventuallyTransducer(Transducer):
    """Eventually (F[a,b]) transducer with proper activation logic"""
    
    def __init__(self, child: Transducer, interval: Tuple[int, int], parent: Optional[Transducer] = None):
        super().__init__(interval=interval, parent=parent)
        self.child = child
        self.children = [child]
        child.parent = self
        self.a, self.b = interval
        self.satisfaction_times = set()  # Track when child was satisfied
    
    def reset(self):
        """Reset transducer to initial state"""
        super().reset()
        self.satisfaction_times.clear()
    
    def trans(self, signal: Union[Tuple[float, float], List[float]], time: float) -> Optional[bool]:
        """Eventually transducer: None before [a,b], active evaluation during, resolved after"""
        child_output = self.child.step(signal, time)
        if child_output is True:
            self.satisfaction_times.add(time)
        
        # Before interval starts: inactive (None)
        if time < self.a:
            self.state = TransducerState.INACTIVE
            return None
        
        # Check if child was satisfied within the active interval [a,b]
        satisfied_in_interval = any(self.a <= t <= self.b for t in self.satisfaction_times)
        
        if satisfied_in_interval:
            # Once satisfied within interval, permanently satisfied
            self.satisfied = True
            self.state = TransducerState.SUCCESS
            return True
        elif time > self.b:
            # After interval ends without satisfaction: permanently violated
            self.violated = True
            self.state = TransducerState.FAILURE
            return False
        else:
            # Within interval [a,b], still evaluating
            self.state = TransducerState.ACTIVE
            return None

class AlwaysTransducer(Transducer):
    """Always (G[a,b]) transducer with proper activation logic"""
    
    def __init__(self, child: Transducer, interval: Tuple[int, int], parent: Optional[Transducer] = None):
        super().__init__(interval=interval, parent=parent)
        self.child = child
        self.children = [child]
        child.parent = self
        self.a, self.b = interval
        self.violation_times = set()  # Track when child was violated
    
    def reset(self):
        """Reset transducer to initial state"""
        super().reset()
        self.violation_times.clear()
    
    def trans(self, signal: Union[Tuple[float, float], List[float]], time: float) -> Optional[bool]:
        """Always transducer: None before [a,b], active evaluation during, resolved after"""
        child_output = self.child.step(signal, time)
        if child_output is False:
            self.violation_times.add(time)
        
        # Before interval starts: inactive (None)
        if time < self.a:
            self.state = TransducerState.INACTIVE
            return None
        
        # Check if child was violated within the active interval [a,b]
        violated_in_interval = any(self.a <= t <= self.b for t in self.violation_times)
        
        if violated_in_interval:
            # Once violated within interval, permanently violated
            self.violated = True
            self.state = TransducerState.FAILURE
            return False
        elif time > self.b:
            # After interval ends without violation: permanently satisfied
            self.satisfied = True
            self.state = TransducerState.SUCCESS
            return True
        else:
            # Within interval [a,b], still evaluating
            self.state = TransducerState.ACTIVE
            return None

class UntilTransducer(Transducer):
    """Until (U[a,b]) transducer with proper activation logic"""
    
    def __init__(self, left: Transducer, right: Transducer, interval: Tuple[int, int], parent: Optional[Transducer] = None):
        super().__init__(interval=interval, parent=parent)
        self.left = left
        self.right = right
        self.children = [left, right]
        left.parent = self
        right.parent = self
        self.a, self.b = interval
        self.right_satisfaction_times = set()
        self.left_violation_times = set()
    
    def reset(self):
        """Reset transducer to initial state"""
        super().reset()
        self.right_satisfaction_times.clear()
        self.left_violation_times.clear()
    
    def trans(self, signal: Union[Tuple[float, float], List[float]], time: float) -> Optional[bool]:
        """Until transducer: φ U[a,b] ψ - φ must hold until ψ becomes true within [a,b]"""
        left_output = self.left.step(signal, time)
        right_output = self.right.step(signal, time)
        
        if right_output is True:
            self.right_satisfaction_times.add(time)
        if left_output is False:
            self.left_violation_times.add(time)
        
        # Before interval starts: inactive (None)
        if time < self.a:
            self.state = TransducerState.INACTIVE
            return None
        
        # Check if right (ψ) was satisfied within [a,b]
        right_satisfied_in_interval = any(self.a <= t <= self.b for t in self.right_satisfaction_times)
        
        if right_satisfied_in_interval:
            # Find earliest satisfaction time of right within interval
            earliest_right_time = min(t for t in self.right_satisfaction_times if self.a <= t <= self.b)
            
            # Check if left was violated before right became true
            left_violated_before_right = any(self.a <= t < earliest_right_time for t in self.left_violation_times)
            
            if not left_violated_before_right:
                # Left held until right became true: permanently satisfied
                self.satisfied = True
                self.state = TransducerState.SUCCESS
                return True
            else:
                # Left was violated before right: permanently violated
                self.violated = True
                self.state = TransducerState.FAILURE
                return False
        elif time > self.b:
            # After interval ends without right satisfaction: permanently violated
            self.violated = True
            self.state = TransducerState.FAILURE
            return False
        else:
            # Within interval [a,b], still evaluating
            self.state = TransducerState.ACTIVE
            return None

def build_transducer_from_tree(tree: Any) -> Transducer:
    """Build transducer from parsed formula tree"""
    if isinstance(tree, str):
        if tree == "false":
            return AtomicTransducer("0>1")  # Always false predicate
        return AtomicTransducer(tree)
    
    op = tree['op']
    if op == '!':
        child = build_transducer_from_tree(tree['arg'])
        return NotTransducer(child)
    elif op == '&':
        left = build_transducer_from_tree(tree['args'][0])
        right = build_transducer_from_tree(tree['args'][1])
        return AndTransducer(left, right)
    elif op == '|':
        left = build_transducer_from_tree(tree['args'][0])
        right = build_transducer_from_tree(tree['args'][1])
        return OrTransducer(left, right)
    elif op == 'F':
        interval = tree['interval']
        child = build_transducer_from_tree(tree['arg'])
        return EventuallyTransducer(child, interval)
    elif op == 'G':
        interval = tree['interval']
        child = build_transducer_from_tree(tree['arg'])
        return AlwaysTransducer(child, interval)
    elif op == 'U':
        interval = tree['interval']
        left = build_transducer_from_tree(tree['args'][0])
        right = build_transducer_from_tree(tree['args'][1])
        return UntilTransducer(left, right, interval)
    else:
        raise ValueError(f"Unsupported operator: {op}")

class STLMonitor:
    """STL Monitor that processes signals through transducers"""
    
    def __init__(self, formula: str):
        self.formula = formula
        self.ast = parse_formula(formula)
        self.transducer = build_transducer_from_tree(self.ast)
    
    def monitor(self, signal: List[List[float]], times: Optional[List[float]] = None) -> List[Optional[bool]]:
        """Monitor signal and return outputs at each time step. Supports continuous time."""
        self.transducer.reset()
        outputs = []
        if times is None:
            # Default: use integer time steps
            times = list(range(len(signal[0])))  # Use length of x coordinates
        
        # Convert signal format: from [[x1,x2,...], [y1,y2,...]] to [(x1,y1), (x2,y2), ...]
        x_coords = signal[0]
        y_coords = signal[1]
        signal_points = list(zip(x_coords, y_coords))
        
        for t, signal_point in zip(times, signal_points):
            output = self.transducer.step(signal_point, t)
            outputs.append(output)
        return outputs
    
    def get_final_verdict(self, signal: List[List[float]], times: Optional[List[float]] = None) -> Optional[bool]:
        """Get final verdict after processing entire signal (supports continuous time)"""
        outputs = self.monitor(signal, times=times)
        for output in reversed(outputs):
            if output is not None:
                return output
        return None
    
    def should_add_to_rrt(self, output: Optional[bool]) -> bool:
        """Determine if an edge should be added to RRT based on transducer output"""
        # ADD edge if output is True or None, REJECT if output is False
        return output is not False

def set_region_params(transducer):
    """Helper function to set region parameters for atomic transducers"""
    if isinstance(transducer, AtomicTransducer):
        if transducer.predicate == 'A':
            transducer.set_region([1.0, 2.0], 1.5)
        elif transducer.predicate == 'B':
            transducer.set_region([3.0, 2.0], 1.0)
    for child in getattr(transducer, 'children', []):
        set_region_params(child)

if __name__ == "__main__":
    print("=== STL Transducer with RRT Integration Logic ===")
    
    # Test case demonstrating the logical system
    formula = "F[1,2](A) & G[2,3](B)"
    monitor = STLMonitor(formula)
    set_region_params(monitor.transducer)
    
    # Test different scenarios
    test_cases = [
        # Time [0,1): Both constraints inactive
        ([0.5, 0.5], 0.5, "Before any constraints activate"),
        ([0.5, 0.5], 0.8, "Still before constraints activate"),
        
        # Time [1,2]: F-constraint active, G-constraint inactive  
        ([1.2, 2.3], 1.2, "F-active: visits A, G-inactive"),
        ([5.0, 0.0], 1.5, "F-active: doesn't visit A, G-inactive"),
        ([1.2, 2.3], 1.8, "F-active: visits A again, G-inactive"),
        
        # Time [2,3]: Both constraints active
        ([3.0, 2.0], 2.2, "Both active: A satisfied, in B"),
        ([5.0, 0.0], 2.5, "Both active: A satisfied, not in B"),
        ([0.0, 0.0], 2.8, "Both active: A satisfied, not in B"),
        
        # Time [3,∞): Both constraints resolved
        ([0.0, 0.0], 3.5, "Both resolved: final state"),
        ([5.0, 5.0], 4.0, "Both resolved: final state"),
    ]
    
    for signal_point, time, description in test_cases:
        monitor.transducer.reset()
        
        # Simulate trajectory up to this point (simplified)
        output = monitor.transducer.step(signal_point, time)
        should_add = monitor.should_add_to_rrt(output)
        
        print(f"Time {time}: {description}")
        print(f"  Signal: {signal_point}, Output: {output}, Add to RRT: {should_add}")
        print(f"  F-satisfied: {monitor.transducer.left.satisfied}, G-violated: {monitor.transducer.right.violated}")
        print()
    
    print("=== Complete Trajectory Test ===")
    # Complete trajectory showing state evolution
    signal = [
        [0.5, 1.2, 3.0, 3.0, 0.0, 0.0],  # A visited at t=1.2, B visited at t=2.2,2.8
        [0.5, 2.3, 2.1, 2.0, 0.0, 0.0]   
    ]
    times = [0.5, 1.2, 2.2, 2.8, 3.5, 4.0]
    
    outputs = monitor.monitor(signal, times=times)
    
    print("Complete trajectory analysis:")
    for i, (t, point, output) in enumerate(zip(times, zip(signal[0], signal[1]), outputs)):
        should_add = monitor.should_add_to_rrt(output)
        print(f"Time {t}: Point {point} -> Output: {output}, Add to RRT: {should_add}")
    
    print(f"\nFinal verdict: {monitor.get_final_verdict(signal, times=times)}")