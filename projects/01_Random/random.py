# File: looping_variance_scene.py
from manim import *
from scipy.stats import *
import numpy as np

class LoopUniform(Scene):
    def construct(self):
        # --- Configuration (Uniform) ---
        WIDTH_MIN = 1.0
        WIDTH_MAX = 8.0
        WIDTH_BASE = 4.0
        amplitude = 3.0
        Y_MAX_HEIGHT_UNIFORM = 1.2 # Max height = 1/WIDTH_MIN = 1.0
        X_RANGE_UNIFORM = [-5, 5]
        Y_RANGE_UNIFORM = [0, Y_MAX_HEIGHT_UNIFORM, 0.2]

        # --- Parameters for ONE OSCILLATION CYCLE ---
        animation_duration = 5
        damping_factor = 0.0
        num_cycles = 1
        frequency = num_cycles * 2 * PI / animation_duration

        # --- Objects (Uniform) ---
        axes = Axes(
            x_range=[*X_RANGE_UNIFORM, 1],
            y_range=Y_RANGE_UNIFORM,
            x_length=10, y_length=6,
            axis_config={"color": WHITE, "include_tip": False},
            x_axis_config={"include_numbers": False},
            y_axis_config={"include_numbers": False},
            tips=False
        )

        width_tracker = ValueTracker(WIDTH_BASE)

        # Helper function to get current params
        def get_uniform_params():
            # Ensure width doesn't go below minimum for calculations
            width = max(WIDTH_MIN, width_tracker.get_value())
            # Prevent division by zero if width somehow becomes zero or negative
            # (shouldn't happen with WIDTH_MIN > 0)
            if width <= 0: width = WIDTH_MIN # Safety check
            loc_a = 0 - width / 2 # Lower bound 'a' (center is 0)
            loc_b = 0 + width / 2 # Upper bound 'b'
            height = 1.0 / width   # PDF height
            return width, loc_a, loc_b, height

        # PDF curve (top horizontal line)
        pdf_curve_uniform = always_redraw(
            lambda: axes.plot(
                lambda x: uniform.pdf(
                    x,
                    loc=get_uniform_params()[1],   # Lower bound 'a'
                    scale=get_uniform_params()[0]  # Width 'w'
                ),
                x_range=X_RANGE_UNIFORM,
                color=RED,
                stroke_width=5,
                discontinuities=[get_uniform_params()[1], get_uniform_params()[2]], # Edges a, b
                use_smoothing=False
            )
        )

        # --- NEW: Add Vertical Bars ---
        # Left vertical bar at x=a
        left_bar = always_redraw(lambda: Line(
                start = axes.c2p(get_uniform_params()[1], 0), # Start at (a, 0)
                end = axes.c2p(get_uniform_params()[1], get_uniform_params()[3]), # End at (a, height)
                color=RED, stroke_width=5
            )
        )

        # Right vertical bar at x=b
        right_bar = always_redraw(lambda: Line(
                start = axes.c2p(get_uniform_params()[2], 0), # Start at (b, 0)
                end = axes.c2p(get_uniform_params()[2], get_uniform_params()[3]), # End at (b, height)
                color=RED, stroke_width=5
            )
        )
        # --- End NEW ---

        # Update function for width (starts decreasing width -> height increases)
        def update_width_uniform(mob, alpha):
            t_prime = alpha * animation_duration
            oscillation = amplitude * np.sin(frequency * t_prime)
            current_width = WIDTH_BASE - oscillation
            current_width = max(WIDTH_MIN, min(WIDTH_MAX, current_width))
            mob.set_value(current_width)

        # --- Construct Scene (Uniform) ---
        # Add axes and all parts of the PDF rectangle
        self.add(axes, pdf_curve_uniform, left_bar, right_bar)

        # Play one oscillation cycle over ~2 seconds
        self.play(
            UpdateFromAlphaFunc(width_tracker, update_width_uniform),
            run_time=animation_duration,
            rate_func=linear
        )
        # End of construct.

class LoopGauss(Scene):
    def construct(self):
        # --- Configuration (mostly the same) ---
        SIGMA_BASE = 1.0
        SIGMA_MIN = 0.25  # Still needed for clamping during oscillation
        SIGMA_MAX = 2.5   # Still needed for clamping
        Y_MAX_HEIGHT = 1 / (SIGMA_MIN * np.sqrt(2 * np.pi)) + 0.2
        X_RANGE = [-5, 5]

        # --- Parameters for UNDAMPED Looping Oscillation ---
        loop_duration = 10       # Duration of one loop cycle
        amplitude = 0.5          # Constant amplitude of oscillation
        # Ensure an integer number of cycles
        num_cycles = 4           # Choose integer number of full cycles for the loop
        frequency = num_cycles * 2 * PI / loop_duration # Angular frequency

        # --- Objects ---
        axes = Axes(
            x_range=[*X_RANGE, 1],
            y_range=[0, Y_MAX_HEIGHT, 0.5],
            x_length=10, y_length=6,
            axis_config={"color": WHITE, "include_tip": False},
            x_axis_config={"include_numbers": False},
            y_axis_config={"include_numbers": False},
            tips=False
        )

        # Start sigma tracker at the beginning state of the loop
        # Since we use "- A * sin(...)", at t=0, sigma = SIGMA_BASE. This is correct.
        sigma_tracker = ValueTracker(SIGMA_BASE)

        pdf_curve = always_redraw(
            lambda: axes.plot(
                lambda x: norm.pdf(
                    x, loc=0, scale=max(SIGMA_MIN, sigma_tracker.get_value())
                ),
                x_range=X_RANGE, color=BLUE_C, stroke_width=5, use_smoothing=True
            )
        )

        # Define the update function for UNDAMPED oscillation
        # Note: t_prime is used conceptually; alpha directly drives the loop.
        def update_sigma_undamped(mob, alpha):
            t_prime = alpha * loop_duration # Time within the loop cycle

            # Calculate the UNDAMPED sine wave component (no exp(-damping * t'))
            # Use the '-' sign to start with sigma decreasing (peak height increasing)
            oscillation = amplitude * np.sin(frequency * t_prime)

            current_sigma = SIGMA_BASE - oscillation

            # Clamp sigma value (still good practice)
            current_sigma = max(SIGMA_MIN, min(SIGMA_MAX, current_sigma))
            mob.set_value(current_sigma)

        # --- Construct the Scene ---
        # Add the static axes and the dynamic curve at its initial state
        self.add(axes, pdf_curve)

        # --- Play ONLY the looping animation ---
        # No initial waits, no final waits, no cleanup plays.
        self.play(
            UpdateFromAlphaFunc(sigma_tracker, update_sigma_undamped),
            run_time=loop_duration,
            rate_func=linear # Linear rate_func is crucial for perfect looping
        )

class LoopStudentT(Scene):
    def construct(self):
        # --- Configuration (Student's t) ---
        DF_MIN = 1        # Minimum allowed df
        DF_MAX = 100       # Maximum allowed df
        DF_BASE = 11.0       # <<<< Baseline MUST be within [DF_MIN, DF_MAX]
        amplitude = 10
        # Adjust Y range? Peak height varies less dramatically than Gaussian variance effect
        # Max height for df=1 is ~0.318, for df=30 is ~0.397.
        Y_MAX_HEIGHT_T = 0.5
        X_RANGE = [-5, 5] # Keep x-range

        # --- Parameters for UNDAMPED Looping Oscillation (Student's t df) ---
        loop_duration = 5      # Duration of one loop cycle

        # Ensure an integer number of cycles
        num_cycles = 2
        frequency = num_cycles * 2 * PI / loop_duration # Angular frequency

        # --- Objects (Student's t) ---
        axes = Axes(
            x_range=[*X_RANGE, 1],
            y_range=[0, Y_MAX_HEIGHT_T, 0.1], # Use adjusted y-range
            x_length=10, y_length=5, # Adjusted y_length maybe
            axis_config={"color": WHITE, "include_tip": False},
            x_axis_config={"include_numbers": False},
            y_axis_config={"include_numbers": False},
            tips=False
        )

        # ValueTracker for degrees of freedom (df)
        df_tracker = ValueTracker(DF_BASE)

        # PDF curve that redraws based on df_tracker
        pdf_curve_t = always_redraw(
            lambda: axes.plot(
                lambda x: t.pdf( # Use t.pdf
                    x,
                    df=max(DF_MIN, df_tracker.get_value()), # Use df from tracker
                    loc=0, # Standard t-distribution has mean 0
                    scale=1 # Standard t-distribution has scale 1
                ),
                x_range=X_RANGE,
                color=BLUE_C, # Different color for distinction
                stroke_width=5,
                use_smoothing=True
            )
        )

        # Update function for df (starts increasing df -> peak increases)
        def update_df_undamped(mob, alpha):
            t_prime = alpha * loop_duration
            # Undamped oscillation term (starts positive)
            oscillation = amplitude * np.sin(frequency * t_prime)
    
            # --- Make df INCREASE first (peak height increases) ---
            current_df = DF_BASE + oscillation
            # Clamp df value (df must be > 0, and maybe cap at max)
            current_df =max(DF_MIN, min(DF_MAX, current_df))
            print(current_df)
            mob.set_value(current_df)

        # --- Construct Scene (Student's t) ---
        self.add(axes, pdf_curve_t) # Add Student's t curve
        self.play(
            UpdateFromAlphaFunc(df_tracker, update_df_undamped), # Use df tracker/updater
            run_time=loop_duration,
            rate_func=linear
        )

class LoopChisq(Scene):
    def construct(self):
        # --- Configuration (Chi-squared) ---
        # Define df range and baseline
        DF_MIN = 3        # Chi2 df must be > 0
        DF_MAX = 30.0        # Upper limit for df
        DF_BASE = 10.0       # Baseline df (must be within [DF_MIN, DF_MAX])
        amplitude = 7.0      # Oscillation amplitude (ensure BASE +/- amp stays within MIN/MAX)
                             # Check: BASE-amp = 10-8=2 >= MIN(1). OK.
                             # Check: BASE+amp = 10+8=18 <= MAX(30). OK.

        # Axis ranges need careful consideration for Chi2
        # Mean=df, Var=2*df. If df max is 18, mean=18, var=36, std=6. Need range >> 18+3*6 = 36
        X_RANGE_CHISQ = [0, 45] # Start at 0, extend right based on max df
        # Peak height is highest for low df (df=1,2). chi2.pdf(0,2)=0.5
        Y_MAX_HEIGHT_CHISQ = 0.5
        Y_RANGE_CHISQ = [0, Y_MAX_HEIGHT_CHISQ, 0.1]

        # --- Parameters for ONE CYCLE UNDAMPED Looping Oscillation ---
        single_cycle_duration = 8 # Duration for one cycle
        damping_factor = 0.0
        num_cycles = 1 # Set exactly ONE cycle for loopability
        frequency = num_cycles * 2 * PI / single_cycle_duration

        # --- Objects (Chi-squared) ---
        axes = Axes(
            x_range=[*X_RANGE_CHISQ, 5], # Use Chi2 x-range, adjust step
            y_range=Y_RANGE_CHISQ,       # Use Chi2 y-range
            x_length=10, y_length=5,     # Adjust lengths as needed
            axis_config={"color": WHITE, "include_tip": False},
            x_axis_config={"include_numbers": False, "decimal_number_config": {"num_decimal_places": 0}}, # Show integer ticks if numbers enabled
            y_axis_config={"include_numbers": False},
            tips=False
        )

        # ValueTracker for degrees of freedom (df)
        df_tracker = ValueTracker(DF_BASE) # Start at baseline

        # PDF curve that redraws based on df_tracker
        pdf_curve_chisq = always_redraw(
            lambda: axes.plot(
                lambda x: chi2.pdf( # Use chi2.pdf
                    x,
                    df=max(DF_MIN, df_tracker.get_value()) # Use df from tracker
                ),
                x_range=X_RANGE_CHISQ, # Use Chi2 range
                color=YELLOW_C,      # Different color
                stroke_width=5,
                use_smoothing=True,
                # Prevent plotting errors for x near 0 when df=1 (where pdf -> inf)
                discontinuities=[0] if df_tracker.get_value() <= 1 else None
            )
        )

        # Update function for df (starts increasing df -> peak moves right/flattens)
        def update_df_chisq(mob, alpha):
            t_prime = alpha * single_cycle_duration
            # Undamped oscillation term (starts positive)
            oscillation = amplitude * np.sin(frequency * t_prime)

            # --- Make df INCREASE first ---
            current_df = DF_BASE + oscillation

            # Clamp df value
            current_df = max(DF_MIN, min(DF_MAX, current_df))
            print(current_df)
            mob.set_value(current_df)

        # --- Construct Scene (Chi-squared) ---
        self.add(axes, pdf_curve_chisq) # Add Chi2 curve
        # Play exactly one cycle
        self.play(
            UpdateFromAlphaFunc(df_tracker, update_df_chisq), # Use df tracker/updater
            run_time=single_cycle_duration,
            rate_func=linear
        )

class LoopExponential(Scene):
    def construct(self):
        # --- Configuration (Exponential) ---
        # Define beta (scale) range and baseline
        BETA_MIN = 0.1         # Min scale (must be > 0) -> Faster decay, Higher peak
        BETA_MAX = 5         # Max scale -> Slower decay, Lower peak
        BETA_BASE = 1.0        # Baseline scale (must be within [MIN, MAX])
        # Amplitude must keep oscillation within bounds
        # Need BASE-amp >= MIN => 1.0-amp >= 0.3 => amp <= 0.7
        # Need BASE+amp <= MAX => 1.0+amp <= 3.0 => amp <= 2.0
        # Max amplitude is 0.7
        amplitude = 0.7

        # Axis ranges for Exponential
        # Mean=beta, StdDev=beta. Max beta is 1.7 here. Mean=1.7, Std=1.7. Need range >> 1.7+3*1.7 = 6.8
        X_RANGE_EXP = [0, 10] # Start at 0, extend right based on max beta in oscillation [0.3, 1.7]
        # Peak height = 1/beta. Max height at BETA_MIN=0.3 is ~3.33.
        Y_MAX_HEIGHT_EXP = 3.5
        Y_RANGE_EXP = [0, Y_MAX_HEIGHT_EXP, 0.5] # Start y at 0

        # --- Parameters for ONE CYCLE UNDAMPED Looping Oscillation ---
        single_cycle_duration = 4.0 # Duration for one cycle
        damping_factor = 0.0
        num_cycles = 1 # Set exactly ONE cycle for loopability
        frequency = num_cycles * 2 * PI / single_cycle_duration

        # --- Objects (Exponential) ---
        axes = Axes(
            x_range=[*X_RANGE_EXP, 1],    # Use Exp x-range
            y_range=Y_RANGE_EXP,          # Use Exp y-range
            x_length=10, y_length=6,      # Adjust lengths as needed
            axis_config={"color": WHITE, "include_tip": False},
            x_axis_config={"include_numbers": False},
            y_axis_config={"include_numbers": False},
            tips=False
        )

        # ValueTracker for scale parameter (beta)
        beta_tracker = ValueTracker(BETA_BASE) # Start at baseline

        # PDF curve that redraws based on beta_tracker
        pdf_curve_exp = always_redraw(
            lambda: axes.plot(
                lambda x: expon.pdf( # Use expon.pdf
                    x,
                    loc=0, # Standard exponential starts at 0
                    scale=max(BETA_MIN, beta_tracker.get_value()) # Use beta from tracker
                ),
                x_range=X_RANGE_EXP, # Use Exp range (starts at 0)
                color=PURPLE,        # Different color
                stroke_width=5,
                use_smoothing=True
            )
        )

        # Update function for beta (starts decreasing beta -> peak increases)
        def update_beta_exponential(mob, alpha):
            t_prime = alpha * single_cycle_duration
            # Undamped oscillation term (starts positive)
            oscillation = amplitude * np.sin(frequency * t_prime)

            # --- Make beta DECREASE first (peak height increases) ---
            current_beta = BETA_BASE - oscillation

            # Clamp beta value
            current_beta = max(BETA_MIN, min(BETA_MAX, current_beta))
            mob.set_value(current_beta)

        # --- Construct Scene (Exponential) ---
        self.add(axes, pdf_curve_exp) # Add Exponential curve
        # Play exactly one cycle
        self.play(
            UpdateFromAlphaFunc(beta_tracker, update_beta_exponential), # Use beta tracker/updater
            run_time=single_cycle_duration,
            rate_func=linear
        )

class LoopGamma(Scene):
    def construct(self):
        # --- Configuration (Gamma - Animating k and theta) ---
        # Define shape (k) range and baseline
        K_MIN = 1.1          # Min shape k
        K_MAX = 15         # Max shape k
        K_BASE = 5.0         # Baseline shape k
        amplitude_k = 3.5      # Oscillation amplitude for k -> Range [1.5, 8.5]

        # Define scale (theta) range and baseline
        THETA_MIN = 0.5        # Min scale theta
        THETA_MAX = 2.0        # Max scale theta
        THETA_BASE = 1.0       # Baseline scale theta
        # Need 1.0-amp_t >= 0.5 => amp_t <= 0.5. Need 1.0+amp_t <= 2.0 => amp_t <= 1.0
        amplitude_theta = 1.4  # Oscillation amplitude for theta -> Range [0.5, 1.5]

        # Axis ranges need to cover extremes of BOTH k and theta
        # Max k=8.5, Max theta=1.5 => Mean~12.75, Var~19.1, Std~4.4. Max extent ~ 13+3*4.4 = 26.2
        X_RANGE_GAMMA = [0, 35] # Start at 0, extend right
        # Max peak height occurs at low k, low theta. k=1.5, theta=0.5 => mode=0.25, height ~1.0
        Y_MAX_HEIGHT_GAMMA = 1.2
        Y_RANGE_GAMMA = [0, Y_MAX_HEIGHT_GAMMA, 0.2] # Start y at 0

        # --- Parameters for ONE CYCLE UNDAMPED Looping Oscillation ---
        single_cycle_duration = 4.0 # Duration for one cycle
        damping_factor = 0.0 # Should be 0 for loop
        num_cycles = 1 # Set exactly ONE cycle for loopability
        frequency = num_cycles * 2 * PI / single_cycle_duration

        # --- Objects (Gamma) ---
        axes = Axes(
            x_range=[*X_RANGE_GAMMA, 5],  # Use wider Gamma x-range
            y_range=Y_RANGE_GAMMA,        # Use Gamma y-range
            x_length=10, y_length=6,      # Adjust lengths as needed
            axis_config={"color": WHITE, "include_tip": False},
            x_axis_config={"include_numbers": False, "decimal_number_config": {"num_decimal_places": 0}},
            y_axis_config={"include_numbers": False},
            tips=False
        )

        # --- ValueTrackers for BOTH parameters ---
        k_tracker = ValueTracker(K_BASE)     # Tracker for shape k
        theta_tracker = ValueTracker(THETA_BASE) # Tracker for scale theta

        # PDF curve that redraws based on BOTH trackers
        pdf_curve_gamma = always_redraw(
            lambda: axes.plot(
                lambda x: gamma.pdf(
                    x,
                    a=max(K_MIN, k_tracker.get_value()),         # Use k from tracker
                    loc=0,
                    scale=max(THETA_MIN, theta_tracker.get_value()) # Use theta from tracker
                ),
                x_range=X_RANGE_GAMMA,
                color=TEAL,
                stroke_width=5,
                use_smoothing=True
                # Add discontinuity handling if k can approach 1 closely?
                # discontinuities=[0] if k_tracker.get_value() <= 1.1 else None
            )
        )

        # Combined update function for k and theta (in phase)
        def update_gamma_params(mob, alpha): # mob is the object the updater is attached to (e.g., k_tracker)
            t_prime = alpha * single_cycle_duration
            # Calculate the single oscillation factor (sine wave from -1 to 1)
            oscillation_factor = np.sin(frequency * t_prime)

            # Update k: Starts INCREASING first
            current_k = K_BASE + amplitude_k * oscillation_factor
            current_k = max(K_MIN, min(K_MAX, current_k)) # Clamp k
            k_tracker.set_value(current_k) # Set k tracker

            # Update theta: Starts INCREASING first (in phase with k)
            current_theta = THETA_BASE + amplitude_theta * oscillation_factor
            current_theta = max(THETA_MIN, min(THETA_MAX, current_theta)) # Clamp theta
            theta_tracker.set_value(current_theta) # Set theta tracker

        # --- Construct Scene (Gamma) ---
        self.add(axes, pdf_curve_gamma) # Add Gamma curve
        # Play exactly one cycle using the combined updater
        # Attach the updater to one of the trackers (it will update both)
        self.play(
            UpdateFromAlphaFunc(k_tracker, update_gamma_params),
            run_time=single_cycle_duration,
            rate_func=linear
        )

class LoopBeta(Scene):
    def construct(self):
        # --- Configuration (Beta - Animating alpha and beta) ---
        # Define alpha and beta range and baseline
        # We want alpha and beta > 0. Let's keep them > 1 for unimodal shapes.
        PARAM_MIN = 1.1          # Min value for alpha and beta
        PARAM_MAX = 15.0         # Max value for alpha and beta
        # Start symmetric
        ALPHA_BASE = 5.0
        BETA_BASE = 5.0
        # Amplitude determines swing range. Ensure BASE +/- amp stays within MIN/MAX
        # Max amplitude = min(BASE - MIN, MAX - BASE) = min(5.0-1.1, 15.0-5.0) = min(3.9, 10.0) = 3.9
        amplitude = 3.5      # Oscillation amplitude -> alpha [1.5, 8.5], beta [1.5, 8.5]

        # Axis ranges for Beta
        X_RANGE_BETA = [0, 1] # Beta is defined on [0, 1]
        # Peak height can get high when one param is low, other high. E.g. Beta(1.5, 8.5)
        # Mode is (a-1)/(a+b-2). Height can exceed 3.
        Y_MAX_HEIGHT_BETA = 4.0
        Y_RANGE_BETA = [0, Y_MAX_HEIGHT_BETA, 0.5] # Start y at 0

        # --- Parameters for ONE CYCLE UNDAMPED Looping Oscillation ---
        single_cycle_duration = 6 # Duration for one cycle
        damping_factor = 0.0
        num_cycles = 1 # Set exactly ONE cycle for loopability
        frequency = num_cycles * 2 * PI / single_cycle_duration

        # --- Objects (Beta) ---
        axes = Axes(
            x_range=[*X_RANGE_BETA, 0.2], # Use Beta x-range [0, 1]
            y_range=Y_RANGE_BETA,         # Use Beta y-range
            x_length=10, y_length=6,      # Adjust lengths as needed
            axis_config={"color": WHITE, "include_tip": False},
            x_axis_config={"include_numbers": False, "decimal_number_config": {"num_decimal_places": 1}},
            y_axis_config={"include_numbers": False},
            tips=False
        )

        # --- ValueTrackers for BOTH parameters ---
        alpha_tracker = ValueTracker(ALPHA_BASE) # Tracker for shape alpha
        beta_tracker = ValueTracker(BETA_BASE)   # Tracker for shape beta

        # PDF curve that redraws based on BOTH trackers
        pdf_curve_beta = always_redraw(
            lambda: axes.plot(
                lambda x: beta.pdf( # Use beta.pdf
                    x,
                    # Use clamped values from trackers
                    a=max(PARAM_MIN, alpha_tracker.get_value()),
                    b=max(PARAM_MIN, beta_tracker.get_value()),
                    loc=0, scale=1 # Standard Beta on [0, 1]
                ),
                x_range=[0.001, 0.999], # Plot slightly inset to avoid potential edge issues
                color=PINK,            # Different color
                stroke_width=5,
                use_smoothing=True
            )
        )

        # Combined update function for alpha and beta (inverse relationship)
        def update_beta_params(mob, alpha): # mob is the object the updater is attached to
            t_prime = alpha * single_cycle_duration
            # Calculate the single oscillation factor (sine wave from -1 to 1)
            oscillation_factor = amplitude * np.sin(frequency * t_prime)

            # Update alpha: Starts INCREASING first
            current_alpha = ALPHA_BASE + oscillation_factor
            current_alpha = max(PARAM_MIN, min(PARAM_MAX, current_alpha)) # Clamp alpha
            alpha_tracker.set_value(current_alpha) # Set alpha tracker

            # Update beta: Starts DECREASING first (INVERSE to alpha)
            current_beta = BETA_BASE - oscillation_factor # NOTE THE MINUS SIGN HERE
            current_beta = max(PARAM_MIN, min(PARAM_MAX, current_beta)) # Clamp beta
            beta_tracker.set_value(current_beta) # Set beta tracker

        # --- Construct Scene (Beta) ---
        self.add(axes, pdf_curve_beta) # Add Beta curve
        # Play exactly one cycle using the combined updater
        # Attach the updater to one of the trackers (it will update both)
        self.play(
            UpdateFromAlphaFunc(alpha_tracker, update_beta_params),
            run_time=single_cycle_duration,
            rate_func=linear
        )

class LoopLogNormal(Scene):
    def construct(self):
        # --- Configuration (LogNormal - Animating sigma) ---
        # --- ADJUSTED PARAMETERS for Higher Peak ---
        SIGMA_LN_MIN = 0.1         # <<<< Lowered Min sigma for higher peak
        SIGMA_LN_MAX = 1.0         # Max sigma (can adjust if needed)
        SIGMA_LN_BASE = 0.5        # Baseline sigma
        # Recalculated Amplitude: max is min(BASE-MIN, MAX-BASE) = min(0.5-0.1, 1.0-0.5) = min(0.4, 0.5) = 0.4
        amplitude = 0.4          # <<<< Adjusted Amplitude -> Range [0.1, 0.9]
        # --- END ADJUSTED PARAMETERS ---
        FIXED_SCALE = 1.0        # Corresponds to mu=0

        # Axis ranges for LogNormal
        X_RANGE_LOGN = [0, 8] # Keep x-range
        # --- ADJUST Y-RANGE for higher peak ---
        Y_MAX_HEIGHT_LOGN = 4.5 # Max peak height is ~4.0 at sigma=0.1
        Y_RANGE_LOGN = [0, Y_MAX_HEIGHT_LOGN, 1] # Adjust step maybe
        # --- END ADJUST Y-RANGE ---

        # --- Parameters for ONE CYCLE UNDAMPED Looping Oscillation ---
        single_cycle_duration = 5 # Duration for one cycle
        damping_factor = 0.0
        num_cycles = 1
        frequency = num_cycles * 2 * PI / single_cycle_duration

        # --- Objects (LogNormal) ---
        axes = Axes(
            x_range=[*X_RANGE_LOGN, 1],    # Use LogN x-range
            y_range=Y_RANGE_LOGN,          # Use NEW LogN y-range
            x_length=10, y_length=6,       # Adjust lengths as needed
            axis_config={"color": WHITE, "include_tip": False},
            x_axis_config={"include_numbers": False, "decimal_number_config": {"num_decimal_places": 0}},
            y_axis_config={"include_numbers": False},
            tips=False
        )

        sigma_ln_tracker = ValueTracker(SIGMA_LN_BASE)

        pdf_curve_lognorm = always_redraw(
            lambda: axes.plot(
                lambda x: lognorm.pdf(
                    x,
                    s=max(SIGMA_LN_MIN, sigma_ln_tracker.get_value()), # Use sigma from tracker
                    loc=0,
                    scale=FIXED_SCALE
                ),
                x_range=[0.01, X_RANGE_LOGN[1]], # Start x slightly > 0
                color=GOLD,
                stroke_width=5,
                use_smoothing=True
            )
        )

        # Update function for sigma (starts decreasing sigma -> less skewed/more peaked)
        # Logic remains the same, uses new BASE and amplitude
        def update_sigma_lognormal(mob, alpha):
            t_prime = alpha * single_cycle_duration
            oscillation = amplitude * np.sin(frequency * t_prime)
            current_sigma_ln = SIGMA_LN_BASE - oscillation
            current_sigma_ln = max(SIGMA_LN_MIN, min(SIGMA_LN_MAX, current_sigma_ln))
            mob.set_value(current_sigma_ln)

        # --- Construct Scene (LogNormal) ---
        self.add(axes, pdf_curve_lognorm)
        self.play(
            UpdateFromAlphaFunc(sigma_ln_tracker, update_sigma_lognormal),
            run_time=single_cycle_duration,
            rate_func=linear # Keep linear rate function for loopability
        )

# --- Custom Rate Function ---
# This function maps t=[0,1] to alpha=[0,1] such that alpha changes
# slowly near t=0, t=0.5, and t=1 (where xi is near 0), and faster
# near t=0.25 and t=0.75 (where xi is near its peaks).
def linger_near_baseline(t):
    return t - np.sin(4 * PI * t) / (4 * PI)

class LoopGEV(Scene):
    def construct(self):
        # --- Configuration (GEV - Animating xi) ---
        MU_GEV = 0.0
        SIGMA_GEV = 1.0
        XI_MIN = -0.6
        XI_MAX = 0.8
        XI_BASE = 0.0
        amplitude = 0.6 # Oscillation amplitude for xi -> Range [-1.0, 1.0]

        # Color settings
        COLOR_FRECHET = BLUE
        COLOR_GUMBEL = GREEN
        COLOR_WEIBULL = RED
        epsilon_inner = 0.10 # Pure Gumbel zone width
        epsilon_outer = 0.30 # Transition end width

        # Axis ranges
        X_RANGE_GEV = [-4, 6]
        Y_MAX_HEIGHT_GEV = 0.7
        Y_RANGE_GEV = [0, Y_MAX_HEIGHT_GEV, 0.1]

        # --- Parameters for ONE CYCLE UNDAMPED Looping Oscillation ---
        single_cycle_duration = 7.0 # Maybe slightly longer duration? Adjust as needed.
        damping_factor = 0.0 # Should be 0
        num_cycles = 1
        frequency = num_cycles * 2 * PI / single_cycle_duration

        # --- Objects (GEV) ---
        axes = Axes(
            x_range=[*X_RANGE_GEV, 1], y_range=Y_RANGE_GEV,
            x_length=10, y_length=6,
            axis_config={"color": WHITE, "include_tip": False},
            x_axis_config={"include_numbers": False, "decimal_number_config": {"num_decimal_places": 1}},
            y_axis_config={"include_numbers": False}, tips=False
        )
        xi_tracker = ValueTracker(XI_BASE)

        pdf_curve_gev = always_redraw(
            lambda: axes.plot(
                lambda x: genextreme.pdf(
                    x, c=-xi_tracker.get_value(),
                    loc=MU_GEV, scale=SIGMA_GEV
                ),
                x_range=[-3.99, X_RANGE_GEV[1]-0.01],
                color=self.get_gev_color_smooth(
                    xi_tracker.get_value(),
                    epsilon_inner, epsilon_outer,
                    COLOR_FRECHET, COLOR_GUMBEL, COLOR_WEIBULL
                    ),
                stroke_width=5, use_smoothing=True,
            )
        )

        # Update function for xi (remains the same logic)
        def update_xi_gev(mob, alpha):
            # Alpha value is now non-linear thanks to rate_func
            t_prime = alpha * single_cycle_duration # Map possibly non-linear alpha to linear time t'
            oscillation = amplitude * np.sin(frequency * t_prime) # Calculate xi based on linear time t'
            current_xi = XI_BASE + oscillation
            current_xi = max(XI_MIN, min(XI_MAX, current_xi))
            mob.set_value(current_xi)

        # --- Construct Scene (GEV) ---
        self.add(axes, pdf_curve_gev)
        self.play(
            UpdateFromAlphaFunc(xi_tracker, update_xi_gev),
            run_time=single_cycle_duration,
            # --- USE THE CUSTOM RATE FUNCTION ---
            rate_func=linger_near_baseline
        )

    # Helper method for SMOOTH dynamic color interpolation (Unchanged)
    def get_gev_color_smooth(self, xi, eps_inner, eps_outer, color_f, color_g, color_w):
        # ... (logic remains the same as previous version) ...
        if xi <= -eps_outer:
            return color_w # Pure Weibull
        elif xi < -eps_inner:
            # Interpolate Weibull -> Gumbel
            alpha = (xi + eps_outer) / (eps_outer - eps_inner)
            return interpolate_color(color_w, color_g, alpha)
        elif xi <= eps_inner:
            return color_g # Pure Gumbel
        elif xi < eps_outer:
            # Interpolate Gumbel -> Frechet
            alpha = (xi - eps_inner) / (eps_outer - eps_inner)
            return interpolate_color(color_g, color_f, alpha)
        else: # xi >= eps_outer
            return color_f # Pure Frechet

class LoopF(Scene):
    def construct(self):
        # --- Configuration (F - Animating d1 and d2) ---
        # Define d1 (dfn) and d2 (dfd) ranges and baselines
        # Keep df > 2 for a defined mode > 0
        D_MIN = 2.1          # Min value for d1 and d2
        D_MAX = 30.0         # Max value for d1 and d2
        # Use same baseline for symmetry in oscillation range
        D1_BASE = 15.0
        D2_BASE = 15.0
        # Amplitude must keep oscillation within bounds [D_MIN, D_MAX]
        # Max amplitude = min(BASE - MIN, MAX - BASE) = min(10.0-2.1, 30.0-10.0) = min(7.9, 20.0) = 7.9
        amplitude = 12      # Oscillation amplitude -> d1/d2 range [3.0, 17.0]

        # Axis ranges for F distribution
        # Varies a lot. Max d1=17, Min d2=3 -> Mean=3. Max d2=17, Min d1=3 -> Mean=1.13
        X_RANGE_F = [0, 6] # Start > 0. Most mass is usually concentrated below ~5-6. Adjust if needed.
        # Peak height varies. Max maybe around 1.3-1.4?
        Y_MAX_HEIGHT_F = 1.5
        Y_RANGE_F = [0, Y_MAX_HEIGHT_F, 0.2] # Start y at 0

        # --- Parameters for ONE CYCLE UNDAMPED Looping Oscillation ---
        single_cycle_duration = 5 # Duration for one cycle
        damping_factor = 0.0
        num_cycles = 1 # Set exactly ONE cycle for loopability
        frequency = num_cycles * 2 * PI / single_cycle_duration

        # --- Objects (F) ---
        axes = Axes(
            x_range=[*X_RANGE_F, 1],    # Use F x-range
            y_range=Y_RANGE_F,          # Use F y-range
            x_length=10, y_length=6,    # Adjust lengths as needed
            axis_config={"color": WHITE, "include_tip": False},
            x_axis_config={"include_numbers": False, "decimal_number_config": {"num_decimal_places": 1}},
            y_axis_config={"include_numbers": False},
            tips=False
        )

        # --- ValueTrackers for BOTH parameters ---
        d1_tracker = ValueTracker(D1_BASE) # Tracker for d1 (dfn)
        d2_tracker = ValueTracker(D2_BASE) # Tracker for d2 (dfd)

        # PDF curve that redraws based on BOTH trackers
        pdf_curve_f = always_redraw(
            lambda: axes.plot(
                lambda x: f.pdf( # Use f.pdf
                    x,
                    # Use clamped values from trackers
                    dfn=max(D_MIN, d1_tracker.get_value()),
                    dfd=max(D_MIN, d2_tracker.get_value()),
                    loc=0, scale=1 # Standard F distribution
                ),
                x_range=[0.01, X_RANGE_F[1]], # Plot slightly > 0
                color=MAROON,           # Different color
                stroke_width=5,
                use_smoothing=True
            )
        )

        # Combined update function for d1 and d2 (inverse relationship)
        def update_f_params(mob, alpha): # mob is the object the updater is attached to
            t_prime = alpha * single_cycle_duration
            # Calculate the single oscillation factor (sine wave from -1 to 1)
            oscillation_factor = amplitude * np.sin(frequency * t_prime)

            # Update d1: Starts INCREASING first
            current_d1 = D1_BASE + oscillation_factor
            current_d1 = max(D_MIN, min(D_MAX, current_d1)) # Clamp d1
            d1_tracker.set_value(current_d1) # Set d1 tracker

            # Update d2: Starts DECREASING first (INVERSE to d1)
            current_d2 = D2_BASE - oscillation_factor # NOTE THE MINUS SIGN HERE
            current_d2 = max(D_MIN, min(D_MAX, current_d2)) # Clamp d2
            d2_tracker.set_value(current_d2) # Set d2 tracker

        # --- Construct Scene (F) ---
        self.add(axes, pdf_curve_f) # Add F curve
        # Play exactly one cycle using the combined updater
        self.play(
            UpdateFromAlphaFunc(d1_tracker, update_f_params), # Attach to one tracker
            run_time=single_cycle_duration,
            rate_func=linear
        )
        # End of construct.

class LoopCauchy(Scene):
    def construct(self):
        # --- Configuration (Cauchy) ---
        # Define gamma (scale) range and baseline
        GAMMA_MIN = 0.3         # Min scale (must be > 0) -> Higher peak
        GAMMA_MAX = 2.0         # Max scale -> Lower peak
        GAMMA_BASE = 1.0        # Baseline scale (must be within [MIN, MAX])
        # Amplitude must keep oscillation within bounds
        # Need BASE-amp >= MIN => 1.0-amp >= 0.3 => amp <= 0.7
        # Need BASE+amp <= MAX => 1.0+amp <= 2.0 => amp <= 1.0
        # Max amplitude is 0.7
        amplitude = 0.7

        # Axis ranges for Cauchy
        # Cauchy tails are heavy, need wider x range than Gaussian/StudentT
        X_RANGE_CAUCHY = [-10, 10]
        # Peak height = 1/(pi*gamma). Max height at GAMMA_MIN=0.3 is ~1.06.
        Y_MAX_HEIGHT_CAUCHY = 1.2
        Y_RANGE_CAUCHY = [0, Y_MAX_HEIGHT_CAUCHY, 0.2] # Start y at 0

        # --- Parameters for ONE CYCLE UNDAMPED Looping Oscillation ---
        single_cycle_duration = 4.0 # Duration for one cycle
        damping_factor = 0.0
        num_cycles = 1 # Set exactly ONE cycle for loopability
        frequency = num_cycles * 2 * PI / single_cycle_duration

        # --- Objects (Cauchy) ---
        axes = Axes(
            x_range=[*X_RANGE_CAUCHY, 2], # Use Cauchy x-range, adjust step
            y_range=Y_RANGE_CAUCHY,       # Use Cauchy y-range
            x_length=12, y_length=6,      # Adjust lengths as needed
            axis_config={"color": WHITE, "include_tip": False},
            x_axis_config={"include_numbers": False},
            y_axis_config={"include_numbers": False},
            tips=False
        )

        # ValueTracker for scale parameter (gamma)
        gamma_tracker = ValueTracker(GAMMA_BASE) # Start at baseline

        # PDF curve that redraws based on gamma_tracker
        pdf_curve_cauchy = always_redraw(
            lambda: axes.plot(
                lambda x: cauchy.pdf( # Use cauchy.pdf
                    x,
                    loc=0, # Center at 0
                    scale=max(GAMMA_MIN, gamma_tracker.get_value()) # Use gamma from tracker
                ),
                x_range=X_RANGE_CAUCHY, # Use Cauchy range
                color=ORANGE,           # Different color
                stroke_width=5,
                use_smoothing=True
            )
        )

        # Update function for gamma (starts decreasing gamma -> peak increases)
        def update_gamma_cauchy(mob, alpha):
            t_prime = alpha * single_cycle_duration
            # Undamped oscillation term (starts positive)
            oscillation = amplitude * np.sin(frequency * t_prime)

            # --- Make gamma DECREASE first (peak height increases) ---
            current_gamma = GAMMA_BASE - oscillation

            # Clamp gamma value
            current_gamma = max(GAMMA_MIN, min(GAMMA_MAX, current_gamma))
            mob.set_value(current_gamma)

        # --- Construct Scene (Cauchy) ---
        self.add(axes, pdf_curve_cauchy) # Add Cauchy curve
        # Play exactly one cycle
        self.play(
            UpdateFromAlphaFunc(gamma_tracker, update_gamma_cauchy), # Use gamma tracker/updater
            run_time=single_cycle_duration,
            rate_func=linear
        )
        # End of construct. The video contains exactly one loopable cycle.


### Testing STUFF
# --- Factory for SINGLE tracker multi-cycle updater ---
def create_updater_factory_for_tracker(param_config, base_value, cycle_dur, total_duration):
    # Extract needed params for THIS tracker
    amp = param_config['amp']
    sign = param_config['sign']
    pmin = param_config['min']
    pmax = param_config['max']
    if cycle_dur <= 0: cycle_dur = 1 # Avoid division by zero
    num_total_cycles = total_duration / cycle_dur
    freq = 1 * 2 * PI / cycle_dur # Ensure num_cycles=1 for frequency calc

    def updater(tracker_mob, alpha): # Now mob IS the tracker
        # Map total alpha (0-1 over total_duration) to alpha within the repeating single cycle
        total_progress_in_cycles = alpha * num_total_cycles
        alpha_in_current_cycle = total_progress_in_cycles % 1.0

        # Calculate param value based on current position in cycle
        t_prime = alpha_in_current_cycle * cycle_dur
        oscillation_base = np.sin(freq * t_prime)
        current_param = base_value + sign * amp * oscillation_base # Use base passed in
        current_param = max(pmin, min(pmax, current_param)) # Clamp
        tracker_mob.set_value(current_param) # Update the tracker directly

    return updater


class LoopingDistribution(VGroup):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config.copy()
        self.trackers = {}
        for param_config in self.config['params']:
            name = param_config['name']
            base = param_config['base']
            self.trackers[name] = ValueTracker(base)

        self.axes = Axes(**self.config['axes_config'])
        # Create INITIAL curve state (using base values)
        self.curve = self._create_curve_object(use_current_tracker=False) # Create plot once
        self.add(self.axes, self.curve)
        # Add persistent updater to morph the curve object itself
        self.curve.add_updater(self._update_curve_mobject)

class LoopingDistribution(VGroup):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config.copy()
        self.trackers = {}
        for param_config in self.config['params']:
            name = param_config['name']
            base = param_config['base']
            self.trackers[name] = ValueTracker(base)

        self.axes = Axes(**self.config['axes_config'])
        # Create INITIAL curve state (using base values)
        self.curve = self._create_curve_object(use_current_tracker=False) # Create plot once
        self.add(self.axes, self.curve)
        # Add persistent updater to morph the curve object itself
        self.curve.add_updater(self._update_curve_mobject)

    # Helper to calculate the dictionary of pdf parameters (for plotting)
    def _get_current_pdf_params(self, use_current_tracker=True):
        pdf_params = {}
        fixed_params = self.config.get('fixed_params', {})
        for key, value in fixed_params.items():
            if value is not None: pdf_params[key] = value

        scipy_param_map = { # Map concept name to SciPy name
             'alpha': 'a', 'beta': 'b', 'k': 'a', 'theta': 'scale','df': 'df',
             'sigma': 'scale', 'sigma_ln': 's', 'gamma': 'scale', 'width': 'scale',
             'xi': 'c', 'd1': 'dfn', 'd2': 'dfd' }
        needs_loc_from_width = False; width_val = None; current_xi_for_color = None;

        for param_config in self.config['params']:
            name = param_config['name']; tracker = self.trackers[name]
            pmin = param_config['min']; pmax = param_config['max']
            base = param_config['base']

            val = tracker.get_value() if use_current_tracker else base
            current_val = max(pmin, min(pmax, val))

            if self.config['name'] == 'GEV' and name == 'xi': current_xi_for_color = current_val

            if self.config['name'] == 'Uniform' and name == 'width':
                pdf_params['scale'] = current_val; width_val = current_val; needs_loc_from_width = True
            elif self.config['name'] == 'GEV' and name == 'xi': pdf_params['c'] = -current_val
            elif name in scipy_param_map: pdf_params[scipy_param_map[name]] = current_val
            else: pdf_params[name] = current_val
        if needs_loc_from_width and width_val is not None: pdf_params['loc'] = 0 - width_val / 2

        # Calculate and store color info separately (used by _create_curve_object)
        current_color = self.config['color']
        if self.config['name'] == 'GEV' and current_xi_for_color is not None:
             if not hasattr(self, 'get_gev_color_smooth'): # Add GEV color method if missing
                 print("Warning: GEV config found but get_gev_color_smooth not defined in LoopingDistribution")
             else:
                 current_color = self.get_gev_color_smooth(
                     current_xi_for_color,
                     self.config['epsilon_inner'], self.config['epsilon_outer'],
                     self.config['color_frechet'], self.config['color_gumbel'], self.config['color_weibull']
                 )
        pdf_params['_custom_color'] = current_color

        return pdf_params

    # Creates the plot object based on parameters - called by init and updater
    def _create_curve_object(self, use_current_tracker=False):
        pdf_params = self._get_current_pdf_params(use_current_tracker=use_current_tracker)
        plot_color = pdf_params.pop('_custom_color', self.config['color']) # Extract color

        # Need a function that uses the calculated params for plotting
        # Define the function here based on the calculated params for this frame
        def pdf_func_for_plot(x):
             # Clamp x slightly away from boundary if needed, esp for Beta
             safe_x = np.clip(x, 0.001, 0.999) if self.config['name']=='Beta' else x
             safe_x = np.clip(safe_x, 0.001, None) if self.config['name']=='LogNormal' else safe_x # Avoid x=0 for LogNormal
             # Add more safety checks?
             try:
                 return self.config['pdf_func'](safe_x, **pdf_params)
             except ValueError: # Catch potential domain errors from SciPy
                 return 0

        return self.axes.plot(
            pdf_func_for_plot, # Use the safer wrapper
            x_range=self.config.get('plot_kwargs', {}).get('x_range', self.axes.x_range[:2]),
            color=plot_color,
            stroke_width=4,
            use_smoothing=True,
            **{k: v for k, v in self.config.get('plot_kwargs', {}).items() if k != 'x_range'}
        )
    
    # Updater function attached to self.curve using add_updater
    def _update_curve_mobject(self, curve_mob, dt=None):
        # Create the target curve shape based on CURRENT tracker values
        target_curve = self._create_curve_object(use_current_tracker=True)
        # Morph the existing curve into the target shape
        curve_mob.become(target_curve)
        return curve_mob # Standard practice for updaters
    
    def get_gev_color_smooth(self, xi, eps_inner, eps_outer, color_f, color_g, color_w):
        # ... (logic remains the same as previous version) ...
        if xi <= -eps_outer: return color_w
        elif xi < -eps_inner:
            alpha = (xi + eps_outer) / (eps_outer - eps_inner); return interpolate_color(color_w, color_g, alpha)
        elif xi <= eps_inner: return color_g
        elif xi < eps_outer:
            alpha = (xi - eps_inner) / (eps_outer - eps_inner); return interpolate_color(color_g, color_f, alpha)
        else: return color_f

    def get_looping_animation(self, total_duration):
        anims = []
        for param_config in self.config['params']:
             tracker = self.trackers[param_config['name']]
             base = param_config['base'] # Need base value for the factory
             # Use the factory that creates an updater for a SINGLE tracker
             updater_f = create_updater_factory_for_tracker(
                 param_config, base, self.config['cycle_duration'], total_duration
             )
             # Apply the update directly to the tracker
             anims.append(UpdateFromAlphaFunc(tracker, updater_f))

        # Return an AnimationGroup that updates all necessary trackers
        # Set lag_ratio=0 so all trackers update based on the same alpha
        return AnimationGroup(*anims, lag_ratio=0)

class SyncLoopTest(Scene):
    def construct(self):
        # ... (Config definition as before) ...
        cauchy_config = {
        'name': 'Cauchy', # Name added for clarity, not used internally yet
        'pdf_func': cauchy.pdf, # SciPy PDF function
        'params': [ # List of parameters to animate
            {
                'name': 'gamma',        # Conceptual parameter name
                'base': 1.0,            # Baseline value for gamma (scale)
                'min': 0.3,             # Minimum allowed value for gamma
                'max': 2.0,             # Maximum allowed value for gamma
                'amp': 0.7,             # Oscillation amplitude (keeps gamma in [0.3, 1.7])
                'sign': -1              # Use BASE - amp*sin() -> gamma decreases first -> peak height increases first
                # 'tracker': None       # Tracker gets added internally by LoopingDistribution
            }
        ],
        'fixed_params': {'loc': 0},    # Fixed SciPy parameters (location)
        'cycle_duration': 1,         # Duration of one oscillation cycle
        'color': ORANGE,               # Color for the curve
        'axes_config': {               # Configuration for Manim Axes
            'x_range': [-10, 10, 2],   # [xmin, xmax, xstep]
            'y_range': [0, 1.2, 0.2],   # [ymin, ymax, ystep]
            'x_length': 6,             # Width on screen
            'y_length': 5              # Height on screen
            # axis_config, x_axis_config, y_axis_config can be added here if needed
            },
        'plot_kwargs': {}              # Extra arguments for axes.plot (like a specific x_range for plotting)
    }
        
        total_duration = 30
        x_shift = 3.5

        dist1 = LoopingDistribution(cauchy_config).shift(LEFT * x_shift)
        dist2 = LoopingDistribution(cauchy_config).shift(RIGHT * x_shift)

        self.add(dist1, dist2)
        self.wait(0.5)

        # Get looping animations that target the TRACKERS
        anim1 = dist1.get_looping_animation(total_duration)
        anim2 = dist2.get_looping_animation(total_duration)

        # Play the animations that update the trackers.
        # The persistent updater on dist1.curve and dist2.curve handles the visuals.
        self.play(
            AnimationGroup(anim1, anim2, lag_ratio=0),
            run_time=total_duration,
            rate_func=linear
        )
        self.wait(1)