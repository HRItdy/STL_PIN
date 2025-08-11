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
        
    def reset(self):
        """Reset transducer to initial state"""
        self.t = 0
        self.state = TransducerState.INACTIVE
        self.state_history.clear()
        self.output_history.clear()
        for child in self.children:
            child.reset()
    
    def step(self, signal: Dict[str, float], time: int) -> Optional[bool]:
        """Process one time step and return output"""
        self.t = time
        output = self.trans(signal, time)
        self.state_history.append(self.state)
        self.output_history.append(output)
        
        return output
    
    @abstractmethod
    def trans(self, signal: Dict[str, float], time: int) -> Optional[bool]:
        """Transition function - must be implemented by subclasses"""
        pass

class AtomicTransducer(Transducer):
    """Atomic predicate transducer"""
    
    def __init__(self, predicate: str, parent: Optional[Transducer] = None):
        super().__init__(parent=parent)
        self.var, self.op, self.threshold = re.split(r'(>=|<=|>|<|==)', predicate)
        self.threshold = float(self.threshold)
        self.predicate = predicate
    
    def trans(self, signal: Dict[str, float], time: int) -> Optional[bool]:
        """Evaluate atomic predicate"""
        val = signal.get(self.var, 0.0)
        result = eval(f"{val} {self.op} {self.threshold}")
        self.state = TransducerState.SUCCESS if result else TransducerState.FAILURE
        return result

class NotTransducer(Transducer):
    """Negation transducer"""
    
    def __init__(self, child: Transducer, parent: Optional[Transducer] = None):
        super().__init__(parent=parent)
        self.child = child
        self.children = [child]
        child.parent = self
    
    def trans(self, signal: Dict[str, float], time: int) -> Optional[bool]:
        """Negate child output"""
        # Always step the child to get current output
        child_output = self.child.step(signal, time)
        
        if child_output is None:
            return None
        
        result = not child_output
        self.state = TransducerState.SUCCESS if result else TransducerState.FAILURE
        return result

class AndTransducer(Transducer):
    """Conjunction transducer"""
    
    def __init__(self, left: Transducer, right: Transducer, parent: Optional[Transducer] = None):
        super().__init__(parent=parent)
        self.left = left
        self.right = right
        self.children = [left, right]
        left.parent = self
        right.parent = self
    
    def trans(self, signal: Dict[str, float], time: int) -> Optional[bool]:
        """Conjunction logic with proper state management"""
        # Always step children to get current outputs
        left_output = self.left.step(signal, time)
        right_output = self.right.step(signal, time)
        
        # Handle None outputs (unknown/pending)
        if left_output is False or right_output is False:
            self.state = TransducerState.FAILURE
            return False
        elif left_output is True and right_output is True:
            self.state = TransducerState.SUCCESS
            return True
        else:
            self.state = TransducerState.ACTIVE
            return None

class OrTransducer(Transducer):
    """Disjunction transducer"""
    
    def __init__(self, left: Transducer, right: Transducer, parent: Optional[Transducer] = None):
        super().__init__(parent=parent)
        self.left = left
        self.right = right
        self.children = [left, right]
        left.parent = self
        right.parent = self
    
    def trans(self, signal: Dict[str, float], time: int) -> Optional[bool]:
        """Disjunction logic with proper state management"""
        # Always step children to get current outputs
        left_output = self.left.step(signal, time)
        right_output = self.right.step(signal, time)
        
        # Handle None outputs (unknown/pending)
        if left_output is True or right_output is True:
            self.state = TransducerState.SUCCESS
            return True
        elif left_output is False and right_output is False:
            self.state = TransducerState.FAILURE
            return False
        else:
            self.state = TransducerState.ACTIVE
            return None

class EventuallyTransducer(Transducer):
    """Eventually (F[a,b]) transducer - 'Once' operator"""
    
    def __init__(self, child: Transducer, interval: Tuple[int, int], parent: Optional[Transducer] = None):
        super().__init__(interval=interval, parent=parent)
        self.child = child
        self.children = [child]
        child.parent = self
        self.a, self.b = interval
        self.satisfaction_times = set()  # Track when child was satisfied
    
    def trans(self, signal: Dict[str, float], time: int) -> Optional[bool]:
        """Eventually transducer based on the 'Once' implementation"""
        # Always step child to get current output
        child_output = self.child.step(signal, time)
        
        # Track satisfaction
        if child_output is True:
            self.satisfaction_times.add(time)
        
        # Check if we have any satisfaction in the interval [time+a, time+b]
        window_start = time + self.a
        window_end = time + self.b
        
        # Check for satisfaction in the current window
        satisfied_in_window = any(window_start <= t <= window_end for t in self.satisfaction_times)
        
        if time < self.a:  # Before window can start
            self.state = TransducerState.INACTIVE
            return None
        elif satisfied_in_window:
            self.state = TransducerState.SUCCESS
            return True
        elif time > self.b:  # Past window
            self.state = TransducerState.FAILURE
            return False
        else:  # In window, still checking
            self.state = TransducerState.ACTIVE
            return None

class AlwaysTransducer(Transducer):
    """Always (G[a,b]) transducer - implemented as !F[a,b]!φ"""
    
    def __init__(self, child: Transducer, interval: Tuple[int, int], parent: Optional[Transducer] = None):
        super().__init__(interval=interval, parent=parent)
        # G[a,b]φ = !F[a,b]!φ
        negated_child = NotTransducer(child)
        self.eventually_transducer = EventuallyTransducer(negated_child, interval)
        self.negation = NotTransducer(self.eventually_transducer)
        self.children = [self.negation]
    
    def trans(self, signal: Dict[str, float], time: int) -> Optional[bool]:
        """Always implemented as double negation of Eventually"""
        return self.negation.trans(signal, time)

class UntilTransducer(Transducer):
    """Until (U[a,b]) transducer"""
    
    def __init__(self, left: Transducer, right: Transducer, interval: Tuple[int, int], parent: Optional[Transducer] = None):
        super().__init__(interval=interval, parent=parent)
        self.left = left
        self.right = right
        self.children = [left, right]
        left.parent = self
        right.parent = self
        self.a, self.b = interval
        self.right_satisfaction_times = set()
    
    def trans(self, signal: Dict[str, float], time: int) -> Optional[bool]:
        """Until transducer implementation"""
        # Always step children to get current outputs
        left_output = self.left.step(signal, time)
        right_output = self.right.step(signal, time)
        
        # Track when right is satisfied
        if right_output is True:
            self.right_satisfaction_times.add(time)
        
        # Check for satisfaction in window [time+a, time+b]
        window_start = time + self.a
        window_end = time + self.b
        
        # Find if right is satisfied in window and left holds until then
        for t_right in self.right_satisfaction_times:
            if window_start <= t_right <= window_end:
                # Check if left held from time to t_right
                left_held = True
                for check_time in range(time, t_right):
                    if check_time < len(self.left.output_history):
                        if self.left.output_history[check_time] is False:
                            left_held = False
                            break
                
                if left_held:
                    self.state = TransducerState.SUCCESS
                    return True
        
        if time < self.a:
            self.state = TransducerState.INACTIVE
            return None
        elif time > self.b:
            self.state = TransducerState.FAILURE
            return False
        else:
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
    
    def monitor(self, signal: List[Dict[str, float]]) -> List[Optional[bool]]:
        """Monitor signal and return outputs at each time step"""
        self.transducer.reset()
        outputs = []
        
        for t, signal_point in enumerate(signal):
            output = self.transducer.step(signal_point, t)
            outputs.append(output)
        
        return outputs
    
    def get_final_verdict(self, signal: List[Dict[str, float]]) -> Optional[bool]:
        """Get final verdict after processing entire signal"""
        outputs = self.monitor(signal)
        # Return the last non-None output, or None if all are None
        for output in reversed(outputs):
            if output is not None:
                return output
        return None
    
if __name__ == "__main__":
    # Example usage
    formula = "F[1,3](x > 5) & G[2,4](y < 10)"
    monitor = STLMonitor(formula)
    
    signal = [
        {'x': 6, 'y': 8},
        {'x': 4, 'y': 9},
        {'x': 7, 'y': 11},
        {'x': 5, 'y': 6},
        {'x': 8, 'y': 12}
    ]
    
    outputs = monitor.monitor(signal)
    print(outputs)  # Outputs at each time step
    final_verdict = monitor.get_final_verdict(signal)
    print(final_verdict)  # Final verdict after processing the signal