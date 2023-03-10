import numpy as np
import torch

from crl.analysis._io import save_json, load_json
from rl_games.analysis.linalg import classify_equilibrium
from crl.models import actor_critic

from loguru import logger

Tensor = torch.Tensor

class FixedPoints(object):
    """Analyze an RNN's dynamics under constant input to find fixed points.

    This class takes an RNN model and finds the location and type of fixed points.
    Inspired by: https://www.mitpressjournals.org/doi/full/10.1162/NECO_a_00409
    "Opening the Black Box: Low-Dimensional Dynamics in High-Dimensional 
    Recurrent Neural Networks" Sussillo and Barak 2013.

    Attributes:
        model: RNN model which is the subject of the fixed point analysis
        speed_tol: speed tolerance used as stopping criteria for fixed point
        dist_th: distance threshold used to distinguish two fixed points
        noise_scale: magnitude of noise added to hidden states
        gamma: step size used for iterating towards fixed points
        fixed_points: array of FixedPoint instances
    """

    def __init__(self,
                 model: actor_critic.RNN,
                 speed_tol: float=1e-1, # float=1e-5
                 dist_th: float=0.15,
                 noise_scale: float=0.2,
    ):
        """Inits FixedPoints class.
        
        Args:
            model: RNN model which is the subject of the fixed point analysis
            speed_tol: speed tolerance used as stopping criteria for fixed point
            dist_th: distance threshold used to distinguish two fixed points
            noise_scale: magnitude of noise added to hidden states
        """
        self.model = model
        self.speed_tol = speed_tol
        self.dist_th = dist_th
        self.noise_scale = noise_scale
        self.gamma = None
        self.fixed_points = None

    def find_fixed_points(self,
                          states,
                          constant_inputs,
                          n_initial_conditions=2,
                          max_iters=5000,
                          gamma=0.02,
    ):
        """Finds fixed points.
        
        Args:
            hidden: RNN hidden state trajectories
            constant_inputs: RNN inputs, which are constant 
            n_initial_conditions: # of samples along trajectory to attempt finding fixed points
            max_iters: maximum # of iterations before giving up on finding fixed point
            gamma: step size used for iterating towards fixed points

        Returns:
            Array of FixedPoint instances
        """
        self.gamma = gamma

        ics = self._get_initial_conditions(states, n_initial_conditions)
        fps = None

        for i, ic in enumerate(ics):
            fp = self._run_initial_condition(
                ic,
                constant_inputs,
                max_iters,
            )

            if fp is not None:
                fps = self._append_fixed_point(fps, fp)

        logger.debug(f"\nFixed Points: {fps}\n")
        
        if fps is not None:
            self.fixed_points = [FixedPoint(n, fp, constant_inputs, self.model) for n, fp in enumerate(fps)]

        return self.fixed_points

    def _get_initial_conditions(self,
                                states: Tensor,
                                n_initial_conditions: int
    ) -> Tensor:
        """Get's set of initial conditions for the analysis.
        Points in input trajectory are sampled and noise is added.
        
        Arguments:
            hidden: trajectory of hidden states 
            n_initial_conditions: number of initial conditions
        
        Returns:
            initial_conditions: initial conditions
        """

        index = np.random.choice(len(states), size=n_initial_conditions, replace=False)
        ics = states[index]

        noise = torch.normal(mean=0, std=self.noise_scale, size=ics.size()).to("cuda")
        ics += noise
                
        ics[0,...] = torch.zeros_like(ics[0,...])

        return ics

    def step_lstm(self, input, s0):
        rank = len(s0.shape)
        n_concat_dims = s0.shape[rank - 1]
        n_dims = n_concat_dims // 2
        c0 = s0[:, n_dims:]
        h0 = s0[:, :n_dims]
        output, (hn, cn) = self.model(input, (h0, c0))
        sn = torch.cat((cn, hn), 1)

        return output, sn


    def _run_initial_condition(self,
                               s, # Tuple[Tensor, Tensor]
                               constant_inputs: Tensor,
                               max_iters: int,
    ) -> Tensor:
        """Iterate initial hidden state to potential fixed point.
        
        Args:
            h: RNN hidden state initial condition
            constant_inputs: RNN inputs, which are constant
            max_iters:
        
        Returns:
            Hidden state corresponding to fixed point or None if fixed point is not found
        """

        # c = ic[0]
        # h = ic[1]
        # c0 = torch.unsqueeze(c, 0)
        # h0 = torch.unsqueeze(h, 0)
        s = torch.unsqueeze(s, 0)
        s.requires_grad = True
        s.retain_grad()

        q_last = 0
        q_last_last = 0

        # s.requires_grad = True
        # s.retain_grad()
        # s0 = (h0, c0)

        gamma = self.gamma
        constant_inputs = torch.unsqueeze(constant_inputs, 0)

        for epoch in range(max_iters):
            # Step though RNN
            # self.model.hx = h
            output, sn = self.step_lstm(constant_inputs[:,epoch,:], s)
            # self.model.update(constant_inputs)
            # states_new = (hn, cn)

            # Compute speed
            q = torch.norm(s.cpu() - sn.cpu())

            # Step
            if q < self.speed_tol:

                # Found a fixed point
                return s
            else:
                # Step in the direction of decreasing speed
                q.backward()

                # Update hidden state, h.grad = dq/dh
                s = s - gamma * s.grad
                s.retain_grad()
                
                # Reduce gamma if q bounces back and forth between two values
                print(q, abs(q - q_last_last))  
                if abs(q - q_last_last) < 1e-5: # TO DO: FIX POINTS BOUNCING BACK AND FORTH AND NOT CONVERGING
                    gamma *= 0.99

                # Update previous speeds
                q_last_last = q_last
                q_last = q
        
        return None

    def _append_fixed_point(self,
                            fps: Tensor,
                            fp: Tensor,
    ) -> Tensor:
        """Append fixed point to list
        
        Args:
            fps:
            fp:
        """
        if fps == None:
            fps = fp
        else:
            dists = torch.linalg.norm(fps - fp, dim=1)
            if torch.min(dists) > self.dist_th:
                fps = torch.cat([fps, fp])
            
        return fps        

    def save_fixed_points(self, filepath):
        """Save fixed points to JSON file."""
        save_json(filepath, [fp.to_dict() for fp in self.fixed_points])

    @staticmethod
    def load_fixed_points(filepath):
        """
        Load fixed points from a .json file
        """
        logger.debug(f"\nLoading fixed points from: {filepath}\n")
        data = load_json(filepath)
        
        return [FixedPoint.from_dict(n, d) for n, d in enumerate(data)]

class FixedPoint(object):
    """Class for computing fixed point type.
    
    This class is responsible taking a fixed point that has already
    been found, and analyzing the Jacobian around it to classify the
    type of fixed point (attractor, repellor, N-saddle).

    Attributes:
        fp_id: fixed point ID
        h: RNN hidden state trajectories
        constant_inputs: RNN inputs, which are constant 
        model: RNN model which is the subject of the fixed point analysis
        dist_th: distance threshold used to distinguish two fixed points
        noise_scale: magnitude of noise added to hidden states
        gamma: step size used for iterating towards fixed points
        fixed_points: array of FixedPoint instances
    """

    def __init__(self,
                 fp_id: int,
                 h: Tensor,
                 constant_input: Tensor,
                 model: actor_critic.RNN=None,
                 jacobian: Tensor=None,
    ):
        """Inits FixedPoint class.

        Args:
            fp_id: fixed point ID
            h: hidden state fixed point of the RNN
            constant_input: constant input into RNN
            model: instance of an RNN class, i.e. RNN model
            eigen_values: Evalues from Jacobian of fixed point
            eigen_vectors: Evectors from Jacobian of fixed point
            jacobian: Jacobian of the dynamics at the fixed point
                If None, the jacobian is computed
            type: type of fixed point (attractor, repellor, N-saddle)
        """
        self.fp_id = fp_id
        self.h = h
        self.constant_input = constant_input
        self.model = model
        self.eigen_values = None
        self.eigen_vectors = None
        self.jacobian = None
        self.type = None

        if jacobian is None:
            self._compute_jacobian()
        else:
            self.jacobian = jacobian

        self._analyze_stability()

    def _compute_jacobian(self):
        """Computes jacobian at the hidden state fixed point of the RNN."""
        n_units = self.h.size(dim=0)
        jacobian = torch.zeros(n_units, n_units)

        # Initialize hidden state
        s0 = self.h
        s0 = torch.unsqueeze(s0, 0)
        constant_input = torch.unsqueeze(self.constant_input, 0)
        # Update RNN model (this updates self.model.hx)
        # self.model.update(self.constant_input)
        output, sn = self.step_lstm(constant_input[:,0,:], s0)
        sn.retain_grad = True
        # Compute Jacobian (change in hidden states from RNN update w.r.t change in individual hidden states)
        for i in range(n_units):
            grad_output = torch.zeros(n_units).to('cuda')
            grad_output = torch.unsqueeze(grad_output, 0)
            grad_output[:,i] = 1

            # self.h: RNN hidden state before update
            # self.model.hx: RNN hidden state after update
            g = torch.autograd.grad(sn, s0, grad_outputs=grad_output, retain_graph=True)[0]
            jacobian[i,:] = g
        
        self.jacobian = jacobian

        logger.debug(f"\nJacobian: {self.jacobian}\n")

    def step_lstm(self, input, s0):
        rank = len(s0.shape)
        n_concat_dims = s0.shape[rank - 1]
        n_dims = n_concat_dims // 2
        c0 = s0[:, n_dims:]
        h0 = s0[:, :n_dims]
        output, (hn, cn) = self.model(input, (h0, c0))
        sn = torch.cat((cn, hn), 1) # GENE NEEDS TO FIX

        return output, sn

    def _analyze_stability(self):
        """
        Analyzes stability around linearized fixed point by inspecting the magnitude 
        of the eigenvalues of the Jacaobian to detect stable/unstable modes
        """
        self.eigen_values, self.eigen_vectors = torch.linalg.eig(self.jacobian)
        self.type = classify_equilibrium(self.eigen_values)


        logger.debug(f"\nEvalues of Jacobian (for classifying fixed point): {self.eigen_values}\n")
        logger.debug(f"\nFixed point type: {self.type}\n")
    
    def to_dict(self):
        """
        Returns the fixed point's attributes as a dictionary,
        used to save FixedPoint to a .json file
        """
        return dict(
            fp_id=self.fp_id,
            h=self.h.tolist(),
            constant_input=self.constant_input.tolist(),
            eigen_values=self.eigen_values.tolist(),
            eigen_vectors=self.eigen_vectors.tolist(),
            jacobian=self.jacobian.tolist(),
            type=self.type,
        )
    
    @classmethod
    def from_dict(cls, fp_id, data_dict):
        """
        Creates an instance of FP from a dictionary of attributes, 
        used when loading FPS from a .json file.
        """
        h = torch.Tensor(data_dict['h']).to('cuda')
        constant_input = torch.Tensor(data_dict['constant_input']).to('cuda')
        jacobian = torch.Tensor(data_dict['jacobian']).to('cuda')
        fp = cls(fp_id, h, constant_input, jacobian=jacobian)
        return fp
    