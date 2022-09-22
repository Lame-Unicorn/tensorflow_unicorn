import tensorflow as tf
from tensorflow.keras.optimizers import schedules


class WarmUpSchedule(schedules.LearningRateSchedule):
    def __init__(self,
                 total_steps,
                 warm_up_steps=None,
                 warm_up_prop=0.1,
                 initial_lr=0.0,
                 warm_up_lr=0.001,
                 end_lr=0.00001,
                 warm_up_pattern='linear',
                 decay_pattern='linear',
                 **kwargs):
        super(WarmUpSchedule, self).__init__(**kwargs)
        self.total_steps = total_steps
        self.warm_up_steps = warm_up_steps
        self._initial_lr = initial_lr
        self._warm_up_lr = warm_up_lr
        self._end_lr = end_lr
        self._warm_up_pattern = warm_up_pattern
        self._decay_pattern = decay_pattern
        if warm_up_steps is None:
            self.warm_up_steps = int(self.total_steps * warm_up_prop)
        else:
            self.warm_up_steps = warm_up_steps

        if end_lr is None:
            end_lr = initial_lr

        self._warm_up_schedule = self.get_interval_schedule(
            initial_lr, self.warm_up_steps, warm_up_lr, warm_up_pattern
        )
        self._decay_schedule = self.get_interval_schedule(
            warm_up_lr, self.total_steps - self.warm_up_steps, end_lr, decay_pattern
        )

    @staticmethod
    def get_interval_schedule(start_lr, steps, end_lr, pattern):
        if pattern == 'linear' or type(pattern) is float:
            if pattern == 'linear':
                power = 1.0
            else:
                power = pattern
            return schedules.PolynomialDecay(
                initial_learning_rate=start_lr,
                decay_steps=steps,
                end_learning_rate=end_lr,
                power=power
            )
        elif pattern == 'exp':
            return schedules.ExponentialDecay(
                initial_learning_rate=start_lr,
                decay_steps=steps,
                decay_rate=end_lr / start_lr,
            )
        else:
            raise ValueError(f"Unknown schedule {pattern}")

    def __call__(self, step):
        # if step < self.warm_up_steps:
        #     return self._warm_up_schedule(step)
        # else:
        #     return self._decay_schedule(step - self.warm_up_steps)
        return tf.cond(step > self.warm_up_steps, lambda: self._warm_up_schedule(step), lambda: self._decay_schedule(step))

    def get_config(self):
        config = super(WarmUpSchedule, self).get_config()
        config.update({
            "total_steps": self.total_steps,
            "warm_up_steps": self.warm_up_steps,
            "initial_lr": self._initial_lr,
            "warm_up_lr": self._warm_up_lr,
            "end_lr": self._end_lr,
            "warm_up_pattern": self._warm_up_pattern,
            "decay_pattern": self._decay_pattern
        })
        return config
