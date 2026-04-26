#!/usr/bin/env python3
"""
1d_FundingRateMeanReversion_WeeklyTrend_v1
Hypothesis: Funding rate mean reversion on 1d with weekly EMA200 trend filter. 
Funding rates extremes indicate overleveraged positions; reversals capture mean reversion moves. 
Weekly EMA200 ensures alignment with major trend (bull/bear) to avoid counter-trend traps. 
Targeting 40-80 total trades over 4 years (10-20/year) to minimize fee drag while capturing funding reversals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Load weekly data ONCE before loop for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema_200 = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1w, ema_200)
    
    # Load funding rate data (assuming it's available as external data)
    # For this experiment, we'll simulate funding rate proxy using price action
    # In practice, replace with actual funding rate data from data/processed/funding/
    returns = np.diff(np.log(close), prepend=0)
    funding_proxy = pd.Series(returns).rolling(window=8, min_periods=8).mean().values  # 8-period proxy for 8h funding
    
    # Z-score of funding proxy (30-period lookback)
    funding_mean = pd.Series(funding_proxy).rolling(window=30, min_periods=30).mean().values
    funding_std = pd.Series(funding_proxy).rolling(window=30, min_periods=30).std().values
    funding_z = np.where(funding_std != 0, (funding_proxy - funding_mean) / funding_std, 0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(30, 200)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_200_aligned[i]) or 
            np.isnan(funding_z[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Weekly trend filter
        uptrend = close[i] > ema_200_aligned[i]
        downtrend = close[i] < ema_200_aligned[i]
        
        # Long signal: funding extremely negative (oversold short positions) + weekly uptrend
        if funding_z[i] < -2.0 and uptrend:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short signal: funding extremely positive (oversold long positions) + weekly downtrend
        elif funding_z[i] > 2.0 and downtrend:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: funding returns to neutral or trend reverses
        elif position == 1 and (funding_z[i] > -0.5 or not uptrend):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (funding_z[i] < 0.5 or not downtrend):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_FundingRateMeanReversion_WeeklyTrend_v1"
timeframe = "1d"
leverage = 1.0