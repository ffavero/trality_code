
from numpy import prod

class BayesianIndicator:

    """
    lookback the specified data for a number of ticks (default 20)
    and compute the probability of the next candle to go up or down
    """

    def __init__(self, n=20):
        self.lookback_period = 20
        self.probabilities = {}

    def add_sequence(self, name, indicator_value, control_value, greater_is_up=True):

        """
        Add indicator by providing the indicator array, and an
        additional control data to compare. The additional data can
        be another array or a number (float/int). 

        """
        if name in self.probabilities:
            raise Exception(
                'the key %s is already beed added to probabilities' % name)
        else:
            self.probabilities[name] = {}
        if isinstance(control_value, (int, float)):
            controls = control_value
        else:
            controls = control_value[-self.lookback_period:]
        seq_up = indicator_value[-self.lookback_period:] > controls
        seq_down = indicator_value[-self.lookback_period:] < controls
        odds_up = sum(seq_up) / self.lookback_period
        odds_down = sum(seq_down) / self.lookback_period
        if greater_is_up:
            self.probabilities[name]['up'] = odds_up / (odds_up + odds_down)
            self.probabilities[name]['down'] = odds_down / (odds_up + odds_down)
        else:
            self.probabilities[name]['up'] = odds_down / (odds_up + odds_down)
            self.probabilities[name]['down'] = odds_up / (odds_up + odds_down)

    def compute_sigma(self):

        p_up = []
        p_down = []
        p_up_not = []
        p_down_not = []
        for key in self.probabilities:
            p_up.append(self.probabilities[key]['up'])
            p_down.append(self.probabilities[key]['down'])
            p_up_not.append(1 - self.probabilities[key]['up'])
            p_down_not.append(1 - self.probabilities[key]['down'])
        prod_up = nan_to_num(prod(p_up))
        prod_down = nan_to_num(prod(p_down))
        prod_up_not = nan_to_num(prod(p_up_not))
        prod_down_not = nan_to_num(prod(p_down_not))
        sigma_up = prod_up / prod_up + prod_up_not
        sigma_down = prod_down / prod_down + prod_down_not
        p_prime = (sigma_up * sigma_down) / (
            sigma_up * sigma_down) + ((1 - sigma_up) * (1 - sigma_down)))
        return(sigma_up, sigma_down, p_prime)

        

def bbbayes(close, bayes_period, bb_upper, bb_basis, sma_values):
    prob_bb_upper_up_seq = greater(close[-bayes_period:],
    bb_upper[-bayes_period:])
    prob_bb_upper_down_seq = less(close[-bayes_period:],
    bb_upper[-bayes_period:])
    prob_bb_basis_up_seq = greater(close[-bayes_period:],
    bb_basis[-bayes_period:])
    prob_bb_basis_down_seq = less(close[-bayes_period:],
    bb_basis[-bayes_period:])
    prob_sma_up_seq = greater(close[-bayes_period:],
    sma_values[-bayes_period:])
    prob_sma_down_seq = less(close[-bayes_period:],
    sma_values[-bayes_period:])
    
    prob_bb_upper_up = sum(
        prob_bb_upper_up_seq) / bayes_period
    prob_bb_upper_down = sum(
        prob_bb_upper_down_seq) / bayes_period
    prob_up_bb_upper = prob_bb_upper_up / (prob_bb_upper_up + prob_bb_upper_down)
    prob_bb_basis_up = sum(
        prob_bb_basis_up_seq) / bayes_period
    prob_bb_basis_down = sum(
        prob_bb_basis_down_seq) / bayes_period
    prob_up_bb_basis = prob_bb_basis_up / (prob_bb_basis_up + prob_bb_basis_down)

    prob_sma_up = sum(
        prob_sma_up_seq) / bayes_period
    prob_sma_down = sum(
        prob_sma_down_seq) / bayes_period
    prob_up_sma = prob_sma_up / (prob_sma_up + prob_sma_down)

    sigma_probs_down = nan_to_num(
        prob_up_bb_upper * prob_up_bb_basis * prob_up_sma / prob_up_bb_upper * prob_up_bb_basis * prob_up_sma + (
            (1 - prob_up_bb_upper) * (1 - prob_up_bb_basis) * (
                1 - prob_up_sma)))
    # Next candles are breaking Up
    prob_down_bb_upper = prob_bb_upper_down / (
        prob_bb_upper_down + prob_bb_upper_up)
    prob_down_bb_basis = prob_bb_basis_down / (
        prob_bb_basis_down + prob_bb_basis_up)
    prob_down_sma = prob_sma_down / (prob_sma_down + prob_sma_up)
    sigma_probs_up = nan_to_num(
        prob_down_bb_upper * prob_down_bb_basis * prob_down_sma / prob_down_bb_upper * prob_down_bb_basis * prob_down_sma + (
            (1 - prob_down_bb_upper) * (1 - prob_down_bb_basis) * (1 - prob_down_sma) ))

    prob_prime = nan_to_num(
        sigma_probs_down * sigma_probs_up / sigma_probs_down * sigma_probs_up + (
            (1 - sigma_probs_down) * (1 - sigma_probs_up)))