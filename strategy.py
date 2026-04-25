#!/usr/bin/env python3
"""
1d Funding Rate Mean Reversion + Weekly Supertrend Filter
Hypothesis: Extreme weekly funding rates predict mean reversion in 1d price action.
In bear markets (2025+), funding often stays negative, creating long bias when extremely negative.
Weekly Supertrend filter ensures we only take mean-reversion trades aligned with the weekly trend
to avoid catching falling knives. Discrete sizing (0.25) minimizes fee churn.
Target: 15-30 trades/year on 1d timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load funding rate data (assumed available as column in prices DataFrame)
    # If not available, this will return zeros and strategy will be inactive
    if 'funding_rate' not in prices.columns:
        return np.zeros(n)
    
    funding = prices['funding_rate'].values
    
    # Get weekly data for Supertrend filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate Weekly Supertrend (10, 3.0)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # ATR calculation
    tr1 = np.abs(np.diff(close_1w, prepend=close_1w[0]))
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr2[0] = np.abs(high_1w[0] - close_1w[0])
    tr3[0] = np.abs(low_1w[0] - close_1w[0])
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = np.zeros(len(close_1w))
    atr_1w[:9] = np.nan
    for i in range(9, len(close_1w)):
        atr_1w[i] = np.mean(tr_1w[i-9:i+1])
    
    # Supertrend calculation
    hl2_1w = (high_1w + low_1w) / 2.0
    upper_1w = hl2_1w + 3.0 * atr_1w
    lower_1w = hl2_1w - 3.0 * atr_1w
    
    supertrend_1w = np.zeros(len(close_1w))
    direction_1w = np.ones(len(close_1w))  # 1 for uptrend, -1 for downtrend
    
    supertrend_1w[0] = upper_1w[0]
    direction_1w[0] = 1
    
    for i in range(1, len(close_1w)):
        if close_1w[i] > supertrend_1w[i-1]:
            direction_1w[i] = 1
        elif close_1w[i] < supertrend_1w[i-1]:
            direction_1w[i] = -1
        else:
            direction_1w[i] = direction_1w[i-1]
        
        if direction_1w[i] == 1:
            supertrend_1w[i] = max(lower_1w[i], supertrend_1w[i-1])
        else:
            supertrend_1w[i] = min(upper_1w[i], supertrend_1w[i-1])
    
    # Align weekly Supertrend direction to 1d timeframe
    supertrend_dir_1w_aligned = align_htf_to_ltf(prices, df_1w, direction_1w)
    
    # Calculate 30-day z-score of funding rate for mean reversion signals
    funding_mean = np.full(n, np.nan)
    funding_std = np.full(n, np.nan)
    for i in range(30, n):
        funding_mean[i] = np.mean(funding[i-29:i+1])
        funding_std[i] = np.std(funding[i-29:i+1])
    
    funding_zscore = np.full(n, np.nan)
    for i in range(30, n):
        if funding_std[i] > 0:
            funding_zscore[i] = (funding[i] - funding_mean[i]) / funding_std[i]
        else:
            funding_zscore[i] = 0.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for funding z-score and Supertrend alignment
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(funding_zscore[i]) or 
            np.isnan(supertrend_dir_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_funding_zscore = funding_zscore[i]
        weekly_trend = supertrend_dir_1w_aligned[i]
        
        if position == 0:
            # Long: extremely negative funding (mean reversion long) AND weekly uptrend
            long_condition = (curr_funding_zscore < -2.0) and (weekly_trend == 1)
            # Short: extremely positive funding (mean reversion short) AND weekly downtrend
            short_condition = (curr_funding_zscore > 2.0) and (weekly_trend == -1)
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: funding normalizes (z-score > -0.5) or weekly trend turns down
            if curr_funding_zscore > -0.5 or weekly_trend == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: funding normalizes (z-score < 0.5) or weekly trend turns up
            if curr_funding_zscore < 0.5 or weekly_trend == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_FundingRateMeanReversion_WeeklySupertrendFilter_v1"
timeframe = "1d"
leverage = 1.0