import tensorflow as tf
from tensorflow.keras.optimizers import Optimizer


class AdamW(Optimizer):
    def __init__(self,
                 learning_rate=0.001,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-7,
                 weight_decay=0.,
                 name='AdamW',
                 **kwargs):
        super(AdamW, self).__init__(name=name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("beta_1", beta_1)
        self._set_hyper("beta_2", beta_2)
        self._set_hyper("weight_decay", weight_decay)
        self.epsilon = epsilon
        if weight_decay > 0.0:
            self._use_weight_decay = True
        else:
            self._use_weight_decay = False

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'm')
            self.add_slot(var, 'v')

    def _resource_apply_dense(self, grad, var):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')
        beta_1 = self._get_hyper("beta_1", var_dtype)
        beta_2 = self._get_hyper("beta_2", var_dtype)
        epsilon = tf.constant(self.epsilon, var_dtype, name="epsilon")
        t = tf.cast(self.iterations + 1, var_dtype)

        m_t = m.assign(
            value=beta_1 * m + (1 - beta_1) * grad,
            use_locking=self._use_locking
        )
        m_hat_t = m_t / (1 - tf.math.pow(beta_1, t))

        v_t = v.assign(
            value=beta_2 * v + (1 - beta_2) * tf.math.pow(grad, 2),
            use_locking=self._use_locking
        )
        v_hat_t = v_t / (1 - tf.math.pow(beta_2, t))

        delta = m_hat_t / (tf.math.sqrt(v_hat_t) + epsilon)

        if self._use_weight_decay:
            weight_decay = self._get_hyper("weight_decay", var_dtype)
            delta += weight_decay * var

        var_updates = var.assign_sub(
            lr_t * delta,
            use_locking=self._use_locking
        )
        return tf.group(var_updates, m_t, v_t)

    def _resource_apply_sparse(self, grad, var, indices):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')
        beta_1 = self._get_hyper("beta_1", var_dtype)
        beta_2 = self._get_hyper("beta_2", var_dtype)
        epsilon = tf.constant(self.epsilon, var_dtype, name="epsilon")
        t = tf.cast(self.iterations + 1, var_dtype)

        m_t = m.assign(
            beta_1 * m,
            use_locking=self._use_locking,
        )
        with tf.control_dependencies([m_t]):
            m_t = m.assign(m.scatter_update(
                resource=m,
                indices=indices,
                updates=(1 - beta_1) * grad,
            ))
        m_hat_t = m_t / (1 - tf.math.pow(beta_1, t))

        v_t = tf.raw_ops.Assign(
            ref=v,
            value=beta_2 * v,
        )
        with tf.control_dependencies([v_t]):
            v_t = v.assign(v.scatter_update(
                resource=v,
                indices=indices,
                updates=(1 - beta_2) * tf.math.pow(grad, 2),
            ))
        v_hat_t = v_t / (1 - tf.math.pow(beta_2, t))

        delta = m_hat_t / (tf.math.sqrt(v_hat_t) + epsilon)

        if self._use_weight_decay:
            weight_decay = self._get_hyper("weight_decay", var_dtype)
            delta += weight_decay * var

        var_updates = tf.raw_ops.AssignSub(
            ref=var,
            value=lr_t * delta,
            use_locking=self._use_locking
        )
        return tf.group(var_updates, m_t, v_t)

    def get_config(self):
        config = super(AdamW, self).get_config()
        config.update({
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "beta_1": self._serialize_hyperparameter("beta_1"),
            "beta_2": self._serialize_hyperparameter("beta_2"),
            "epsilon": self.epsilon,
            "weight_decay": self._serialize_hyperparameter("weight_decay"),
        })
        return config
