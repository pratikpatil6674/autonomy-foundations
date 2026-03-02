import casadi as ca
import numpy as np
from reference_trajectories import sample_reference_trajectory
import do_mpc


class MPC:
    def __init__(self, vehicle_params, sim_params, reference_trajectory):
        self.vehicle_params = vehicle_params
        self.sim_params = sim_params
        self.reference_trajectory = reference_trajectory
        self.model, self.mpc = self.setup_model()
        
    def setup_model(self):
        model = do_mpc.model.Model("discrete")

        # states
        x = model.set_variable(var_type='states', var_name='x', shape=(1,1))
        y = model.set_variable(var_type='states', var_name='y', shape=(1,1))
        psi = model.set_variable(var_type='states', var_name='psi', shape=(1,1))
        v = model.set_variable(var_type='states', var_name='v', shape=(1,1))
        delta = model.set_variable(var_type='states', var_name='delta', shape=(1,1))
        
        # inputs
        a = model.set_variable(var_type='_u', var_name='a', shape=(1,1))
        delta_cmd = model.set_variable(var_type='_u', var_name='delta_cmd', shape=(1,1))
        
        # TVP reference for each step in horizon (we’ll fill these in each solve)
        Xr   = model.set_variable('_tvp', 'Xr')
        Yr   = model.set_variable('_tvp', 'Yr')
        psir = model.set_variable('_tvp', 'psir')
        vr   = model.set_variable('_tvp', 'vr')
        deltar   = model.set_variable('_tvp', 'deltar')
        
        # discrete dynamics (Euler integration)
        X_next = x + v * ca.cos(psi) * self.sim_params.dt
        Y_next = y + v * ca.sin(psi) * self.sim_params.dt
        Psi_next = psi + v * ca.tan(delta) / self.vehicle_params.wheelbase * self.sim_params.dt
        Psi_next = ca.arctan2(ca.sin(Psi_next), ca.cos(Psi_next))   # wrap heading angle to [-pi, pi]
        V_next = v + a * self.sim_params.dt
        delta_next = delta + (delta_cmd - delta) / self.vehicle_params.steering_constant * self.sim_params.dt
        
        model.set_rhs('x', X_next)
        model.set_rhs('y', Y_next)
        model.set_rhs('psi', Psi_next)
        model.set_rhs('v', V_next)
        model.set_rhs('delta', delta_next)
        
        model.setup()

        mpc = do_mpc.controller.MPC(model)

        self.setup_mpc = {
        'n_horizon': 60,
        't_step': self.sim_params.dt,
        'state_discretization': 'discrete',
        'store_full_solution': True,
        'nlpsol_opts': {'ipopt.print_level': 0, 'print_time': 0}
        }

        mpc.set_param(**self.setup_mpc)

        # Objective: tracking + smoothness
        # Weights (tune)
        w_pos = 10.0
        w_psi = 2.0
        w_v   = 1.0
        w_delta = 2.0
        w_u   = 0.1
        w_du  = 1.0

        # Stage cost, lagrange term
        lterm = (
            w_pos * ((x - Xr)**2 + (y - Yr)**2)
            # w_psi * (psi - psir)**2 +
            # w_v * (v - vr)**2 +
            # w_delta * (delta - deltar)**2
        )
        mpc.set_objective(mterm=lterm, lterm=lterm)

        # Penalize input changes (smoothness)
        mpc.set_rterm(a=w_u, delta_cmd=w_du)

        # Constraints
        mpc.bounds['lower','_u','a'] = self.vehicle_params.a_min
        mpc.bounds['upper','_u','a'] = self.vehicle_params.a_max
        mpc.bounds['lower','_u','delta_cmd'] = self.vehicle_params.delta_min
        mpc.bounds['upper','_u','delta_cmd'] = self.vehicle_params.delta_max

        mpc.bounds['lower','_x','delta'] = self.vehicle_params.delta_min
        mpc.bounds['upper','_x','delta'] = self.vehicle_params.delta_max
        mpc.bounds['lower','_x','v'] = 0.0

        # --- TVP function: fill reference at each horizon step ---
        tvp_template = mpc.get_tvp_template()

        def tvp_fun(t_now):
            t_now_float = to_float_time(t_now)
            for k in range(self.setup_mpc['n_horizon'] + 1):
                tq = t_now_float + k * self.setup_mpc['t_step']
                Xr, Yr, psir, vr, deltar = sample_reference_trajectory(self.reference_trajectory, tq)
                tvp_template['_tvp', k, 'Xr'] = Xr
                tvp_template['_tvp', k, 'Yr'] = Yr
                tvp_template['_tvp', k, 'psir'] = psir
                tvp_template['_tvp', k, 'vr'] = vr
                tvp_template['_tvp', k, 'deltar'] = deltar
            return tvp_template

        mpc.set_tvp_fun(tvp_fun)

        mpc.setup()
        return model, mpc
    
    def build_simulator(self):
        simulator = do_mpc.simulator.Simulator(self.model)
        simulator.set_param(t_step=self.sim_params.dt)
        # If you used TVPs in model, you must define them for simulator too.
        tvp_template = simulator.get_tvp_template()
        def tvp_fun_sim(t_now):
            # Plant doesn't need reference; keep TVPs defined (or mirror MPC)
            # t_now_float = to_float_time(t_now)
            # for k in range(self.setup_mpc['n_horizon'] + 1):
            #     tq = t_now_float + k * self.setup_mpc['t_step']
            #     Xr, Yr, psir, vr, deltar = sample_reference_trajectory(self.reference_trajectory, tq)
            #     tvp_template['_tvp', k, 'Xr'] = Xr
            #     tvp_template['_tvp', k, 'Yr'] = Yr
            #     tvp_template['_tvp', k, 'psir'] = psir
            #     tvp_template['_tvp', k, 'vr'] = vr
            #     tvp_template['_tvp', k, 'deltar'] = deltar
            return tvp_template
        simulator.set_tvp_fun(tvp_fun_sim)
        simulator.setup()
        self.simulator = simulator
    
    def run_closed_loop(
        self, 
        x0, 
        steps = 1000
    ):
        self.mpc.x0 = x0
        self.simulator.x0 = x0
        self.mpc.set_initial_guess()
        
        xk = x0
        states = []
        inputs = []
        # Run closed-loop simulation
        for _ in range(steps):
            uk = self.mpc.make_step(xk)
            xk = self.simulator.make_step(uk)
            states.append(xk)
            inputs.append(uk)
        
        return states, inputs

def to_float_time(t_now) -> float:
    # Handles int/float, numpy scalars/arrays, casadi DM, lists
    if isinstance(t_now, (int, float)):
        return float(t_now)

    # numpy scalar or array
    if isinstance(t_now, np.ndarray):
        return float(np.asarray(t_now).reshape(-1)[0])

    # casadi types often have .full() or can be cast via np.array(...)
    try:
        return float(np.array(t_now).reshape(-1)[0])
    except Exception:
        # last resort
        return float(t_now)