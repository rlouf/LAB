""" Conversion models """

# Authors: Remi Louf <remilouf@gmail.com>
#          Etienne Duschene <etienne.duschene@ens-lyon.org>


import numpy as np
import pymc3 as pm


__all__ = ['Conversion']


# Needed:
#  - Improvement distribution
#  - Difference distribution
#  - Plot differences
#  - Confidence that it is better
#  - Loss function goes in something more "Meta"
        
class Conversion():
    """ Conversion model. 

    We assume that the number of successful transitions between two states is
    obtained via a Binomial process where the probability of success is the
    `true' conversion rate :eq:`\theta`

    .. math::
        
        successes ~ Binomial(trials, \theta)

    We want to infer this conversion rate based on empirical measurements.
    
    ## Multiple variants

    When performing a test with multiple variants, we need to compensate for the
    fact that the more variants we have, the more likely we are to have false
    positives. 
    
    We cannot simply consider all variants in isolation when performing our
    measures. Intuitively, if the conversion rate of variant A is 1% there is
    little chance that the conversion rate of variant B would be 90%. We thus
    need to take all available data into account.
    
    They key here is to consider that variants are exchangeable, i.e. you have no specific information
    that would be able to distinguish between experiments. It turns out that
    when the variants are infinitely exchangeable, we can show that there
    exists a variable :eq:`\beta` that makes all the observation
    conditionally independent.  We assume, following [Gellman] and [Stich] that
    all the 'true' rates share the same beta prior (which is not too crazy
    since the beta distribution can approximate almost any distribution on
    [0,1]):

    .. math::
        
        true_rate ~ Beta(a, b)

    In the absence of other information, we give an uninformative prior to the
    values of :eq:`(a, b)` which takes the following form (cf [Gelman]):

    .. math:: f(a,b) \propto 1/(a+b)^{5/2}

    Following [Stan] we reparametrize this prior to facilitate the inference:
    
     .. math::

            a &= \phi \mult \lambda\\
            b &= \lambda \mult (1- \phi)\\
            \phi &\propto Uniform(0,1)\\
            \lambda &\propto Pareto(1.5, 0.1)\\

    Example
    -------
    >>> import lab
    >>> import numpy as np
    >>> data = [[100,10], [103, 56], [120, 23]]
    >>> conversion = lab.Conversion()
    >>> conversion.fit(data)
    
    References
    ----------
    .. [Gelmann] Bayesian Data Analysis
    .. [Stich] http://sl8r000.github.io/ab_testing_statistics/use_a_hierarchical_model/
    .. [Stan] https://github.com/stan-dev/stan/releases/download/v2.17.0/stan-reference-2.17.0.pdf  p. 285
    """

    def fit(self, data):
        """ Fit the conversion model
        """
        self.model = self.create_model()
        with self.model:
            self.trace = pm.sample()

    def create_model(self, X):
        """ Implement the conversion model with PyMC3.

        See class documentation for explanations.
        """

        self.nb_variants = len(X)

        with pm.Model() as model:
            phi = pm.Uniform('phi', 0, 1, testval=0.5)
            lmbda = pm.Pareto('lambda', 1.5, 0.1, testval=1)

            # Shared Prior for the true rates
            true_rates = pm.Beta('p', phi*lmbda, lmbda*(1-phi), shape=self.nb_variants)

            # The observed rates
            for i, variant in enumerate(X):
                pm.Binomial('observed_values_{}'.format(i),
                            variant[0],
                            true_rates[i],
                            observed=variant[1])

        self.named_vars = {'theta': true_rates}
        self.model = model

    return model

    def save():
        raise NotImplementedError
    
    def load():
        raise NotImplementedError
