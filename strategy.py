#/usr/bin/env python3
# 1d_Wilcoxon_Rank_Sum_Trend
# Hypothesis: Use Wilcoxon rank-sum test on 20-day returns to detect regime shifts. 
# Long when recent 10-day returns significantly outperform prior 10-day (p<0.05) and price > weekly VWAP.
# Short when recent 10-day significantly underperform (p<0.05) and price < weekly VWAP.
# Weekly trend filter prevents counter-trend trades. Low frequency due to statistical test requirement.
# Works in bull (captures sustained momentum) and bear (identifies distribution/accumulation phases).

name = "1d_Wilcoxon_Rank_Sum_Trend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from scipy import stats
from mtf_data import get_htf_data, align_htf_to_ltf

def wilcoxon_p_value(x, y):
    """Compute p-value for Wilcoxon rank-sum test (Mann-Whitney U)"""
    if len(x) < 2 or len(y) < 2:
        return 1.0
    try:
        _, p_value = stats.ranksums(x, y)
        return p_value
    except:
        return 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get weekly data for trend and VWAP
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly VWAP: typical price * volume / cumulative volume
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    vwap = (typical_price * df_1w['volume']).cumsum() / df_1w['volume'].cumsum()
    vwap_values = vwap.values
    
    # Daily returns for Wilcoxon test
    returns = np.diff(np.log(close), prepend=0)
    
    # Align weekly VWAP to daily
    vwap_aligned = align_htf_to_ltf(prices, df_1w, vwap_values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):  # Need 20 days for test + 10 buffer
        # Skip if any required value is NaN
        if np.isnan(vwap_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if i >= 30:  # Need 20 days of returns for test
            # Split last 20 days into two 10-day periods
            recent_returns = returns[i-10:i]   # Last 10 days
            prior_returns = returns[i-20:i-10] # Prior 10 days
            
            # Skip if insufficient data
            if len(recent_returns) < 10 or len(prior_returns) < 10:
                if position != 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.0
                continue
                
            # Wilcoxon rank-sum test
            p_value = wilcoxon_p_value(recent_returns, prior_returns)
            
            if position == 0:
                # LONG: recent significantly better than prior (p<0.05) and price > weekly VWAP
                if p_value < 0.05 and np.median(recent_returns) > np.median(prior_returns) and close[i] > vwap_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # SHORT: recent significantly worse than prior (p<0.05) and price < weekly VWAP
                elif p_value < 0.05 and np.median(recent_returns) < np.median(prior_returns) and close[i] < vwap_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            elif position == 1:
                # EXIT LONG: trend weakening or price below VWAP
                if p_value >= 0.05 or close[i] < vwap_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # EXIT SHORT: trend strengthening or price above VWAP
                if p_value >= 0.05 or close[i] > vwap_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
        else:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0

    return signals