from spaceship_environment import SpaceshipEnvironment
from imagination_based_planner import ImaginationBasedPlanner


def _1_test_imaginator_dummy_action_0():
    experiment_name = "test_imaginator_dummy_action_0"

    environment = SpaceshipEnvironment(n_planets=3, render_after_each_step=True)
    agent = ImaginationBasedPlanner(environment, experiment_name=experiment_name, tensorboard=False, max_imaginations_per_action=0, n_episodes_per_batch=500, train=False,
                                    use_controller_and_memory=False, dummy_action_magnitude_interval=(0, 0), use_ship_mass=True)
    agent.load(experiment_name)

    for i_episode in range(14000):
        environment.reset()

        for i_action in range(environment.n_actions_per_episode):
            agent.act()

        agent.finish_episode()


def _2_controlmem_batch500_2imag_default_lr():
    experiment_name = "controlmem_batch500_2imag_default-lr"

    environment = SpaceshipEnvironment(n_planets=3, render_after_each_step=True)
    agent = ImaginationBasedPlanner(environment, experiment_name=experiment_name, tensorboard=False, max_imaginations_per_action=2, n_episodes_per_batch=500, train=False,
                                    use_controller_and_memory=True, dummy_action_magnitude_interval=(0, 0), use_ship_mass=True, erroneous_version=True)
    agent.load(experiment_name)

    for i_episode in range(14000):
        environment.reset()

        for i_action in range(environment.n_actions_per_episode):
            agent.act()

        agent.finish_episode()


def _3_test_imaginator_dummy_action_0_8_actnorm_0_5_lr_0_001_batch_100():
    experiment_name = "test_imaginator_dummy-action-0-8_actnorm-0.5_lr-0.001_batch-100"

    environment = SpaceshipEnvironment(n_planets=3, render_after_each_step=True)
    agent = ImaginationBasedPlanner(environment, experiment_name=experiment_name, tensorboard=False, max_imaginations_per_action=0, n_episodes_per_batch=500, train=False,
                                    use_controller_and_memory=False, dummy_action_magnitude_interval=(0, 8), use_ship_mass=True)
    agent.imaginator.action_normalization_factor = 0.5
    agent.load(experiment_name)

    for i_episode in range(14000):
        environment.reset()

        for i_action in range(environment.n_actions_per_episode):
            agent.act()

        agent.finish_episode()


def _4_test_imaginator_dummy_action_0_10_actnorm_0_5_lr_0_001_batch_100_reg0_no_mass_0_25():
    experiment_name = "test_imaginator_dummy-action-0-10_actnorm-0.5_lr-0.001_batch-100_reg0_no-mass-0.25"

    environment = SpaceshipEnvironment(n_planets=3, render_after_each_step=True, agent_ship_random_mass_interval=(0.25, 0.25))
    agent = ImaginationBasedPlanner(environment, experiment_name=experiment_name, tensorboard=False, max_imaginations_per_action=0, n_episodes_per_batch=500, train=False,
                                    use_controller_and_memory=False, dummy_action_magnitude_interval=(0, 10), use_ship_mass=False)
    agent.imaginator.action_normalization_factor = 0.5
    agent.load(experiment_name)

    for i_episode in range(14000):
        environment.reset()

        for i_action in range(environment.n_actions_per_episode):
            agent.act()

        agent.finish_episode()


def _5_full_system_pre_trained_batch200_imag_2_EMlr0_001_others_default_no_reg_fixedmass_0_25():
    experiment_name = "full-system_pre-trained_batch200_imag-2_EMlr0.001_others-default_no-reg_fixedmass-0.25"

    environment = SpaceshipEnvironment(n_planets=3, render_after_each_step=True, agent_ship_random_mass_interval=(0.25, 0.25))
    agent = ImaginationBasedPlanner(environment, experiment_name=experiment_name, tensorboard=False, max_imaginations_per_action=2, n_episodes_per_batch=500, train=False,
                                    use_controller_and_memory=True, dummy_action_magnitude_interval=(0, 10), use_ship_mass=True, erroneous_version=True)
    agent.load(experiment_name)

    for i_episode in range(14000):
        environment.reset()

        for i_action in range(environment.n_actions_per_episode):
            agent.act()

        agent.finish_episode()


def _6_full_system_bootstrapped_batch200_imag_2_EMlr0_001_others_default_no_reg_no_mass_0_25():
    experiment_name = "full-system_bootstrapped_batch200_imag-2_EMlr0.001_others-default_no-reg_no-mass-0.25"

    environment = SpaceshipEnvironment(n_planets=3, render_after_each_step=True, agent_ship_random_mass_interval=(0.25, 0.25))
    agent = ImaginationBasedPlanner(environment, experiment_name=experiment_name, tensorboard=False, max_imaginations_per_action=2, n_episodes_per_batch=500, train=False,
                                    use_controller_and_memory=True, dummy_action_magnitude_interval=(0, 10), use_ship_mass=False, erroneous_version=True)
    agent.load(experiment_name)

    for i_episode in range(14000):
        environment.reset()

        for i_action in range(environment.n_actions_per_episode):
            agent.act()

        agent.finish_episode()


def _7_full_system_bootstrapped_batch200_imag_2_EMlr0_001_others_default_no_reg_no_mass_0_25_fuel_0_0004_refresh():
    experiment_name = "full-system_bootstrapped_batch200_imag-2_EMlr0.001_others-default_no-reg_no-mass-0.25_fuel-0.0004_refresh"

    environment = SpaceshipEnvironment(n_planets=3, render_after_each_step=True, agent_ship_random_mass_interval=(0.25, 0.25))
    agent = ImaginationBasedPlanner(environment, experiment_name=experiment_name, tensorboard=False, max_imaginations_per_action=2, n_episodes_per_batch=500, train=False,
                                    use_controller_and_memory=True, dummy_action_magnitude_interval=(0, 10), use_ship_mass=False, erroneous_version=True)
    agent.load(experiment_name)

    for i_episode in range(14000):
        environment.reset()

        for i_action in range(environment.n_actions_per_episode):
            agent.act()

        agent.finish_episode()



def _8_new_system_bootstrapped_batch250_imag_2_EMlr0_001_others_default_no_reg_no_mass_0_25_fuel_0_0004():
    experiment_name = "new-system_bootstrapped_batch250_imag-2_EMlr0.001_others-default_no-reg_no-mass-0.25_fuel-0.0004"

    environment = SpaceshipEnvironment(n_planets=3, render_after_each_step=True, agent_ship_random_mass_interval=(0.25, 0.25))
    agent = ImaginationBasedPlanner(environment, experiment_name=experiment_name, tensorboard=False, max_imaginations_per_action=2, n_episodes_per_batch=500, train=False,
                                    use_controller_and_memory=True, dummy_action_magnitude_interval=(0, 10), use_ship_mass=False, erroneous_version=False)
    agent.load(experiment_name)

    for i_episode in range(14000):
        environment.reset()

        for i_action in range(environment.n_actions_per_episode):
            agent.act()

        agent.finish_episode()



# _1_test_imaginator_dummy_action_0()
# _2_controlmem_batch500_2imag_default_lr()
# _3_test_imaginator_dummy_action_0_8_actnorm_0_5_lr_0_001_batch_100()
# _4_test_imaginator_dummy_action_0_10_actnorm_0_5_lr_0_001_batch_100_reg0_no_mass_0_25()
# _5_full_system_pre_trained_batch200_imag_2_EMlr0_001_others_default_no_reg_fixedmass_0_25()
# _6_full_system_bootstrapped_batch200_imag_2_EMlr0_001_others_default_no_reg_no_mass_0_25()
# _7_full_system_bootstrapped_batch200_imag_2_EMlr0_001_others_default_no_reg_no_mass_0_25_fuel_0_0004_refresh()
_8_new_system_bootstrapped_batch250_imag_2_EMlr0_001_others_default_no_reg_no_mass_0_25_fuel_0_0004()