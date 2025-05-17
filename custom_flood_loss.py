import tensorflow as tf
def custom_flood_loss(y_true, y_pred):
    """
    Custom loss function for flood forecasting.

    This function penalizes prediction errors during both flood and non-flood events,
    with separate weights. The aim is to reduce the overall prediction error while still
    prioritizing accuracy during critical flood periods. A smoothness regularization 
    term is also included to discourage sudden changes in predicted values.

    Parameters:
    - y_true: Tensor of true streamflow values.
    - y_pred: Tensor of predicted streamflow values.
    - flood_threshold: Threshold above which streamflow is considered flooding.
    - alpha_flood: Penalty weight for errors during flood events.
    - alpha_non_flood: Penalty weight for errors during non-flood events.
    - beta: Weight for regularization promoting prediction smoothness.

    Returns:
    - Total loss combining flood, non-flood, and smoothness penalties.
    """

    # Define domain-specific parameters 
    flood_threshold = 0.0153683  #   Threshold selected by GPD method
    alpha1 = 0.2 # Weight for penalizing errors during flood events
    alpha2 = 0.4 # Weight for penalizing errors during non flood events

    beta = 0.1  # Weight for regularization term

    # Calculate absolute errors
    abs_errors = tf.abs(y_true - y_pred)

    # Separate errors during flood events and non-flood events
    flood_errors = tf.where(y_true > flood_threshold, abs_errors, tf.zeros_like(abs_errors))
    non_flood_errors = tf.where(y_true <= flood_threshold, abs_errors, tf.zeros_like(abs_errors))

    # Penalize errors during flood events more heavily
    flood_loss = tf.reduce_mean(flood_errors) * alpha1
    
    # Penalize errors during non_flood events more heavily
    non_flood_loss = tf.reduce_mean(non_flood_errors) * alpha2
    
    # Regularization term to encourage smoothness in predicted flow rates (L2 regularization)
        
    regularization_term = beta * tf.reduce_mean(tf.square(y_pred[:, 1:] - y_pred[:, :-1]))

    # Total loss
    loss = flood_loss + regularization_term + non_flood_loss 
    
    return loss
