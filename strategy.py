#!/usr/bin/env python3
name = "1d_1w_PriceAction_Breakout"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w Highest High and Lowest Low over 20 periods
    hh = pd.Series(df_1w['high'].values).rolling(window=20, min_periods=20).max().values
    ll = pd.Series(df_1w['low'].values).rolling(window=20, min_periods=20).min().values
    hh_ll_1w = align_htf_to_ltf(prices, df_1w, hh)
    ll_ll_1w = align_htf_to_ltf(prices, df_1w, ll)
    
    # Volume spike detection (20-period average on daily)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for 20-period HH/LL
    
    for i in range(start_idx, n):
        if np.isnan(hh_ll_1w[i]) or np.isnan(ll_ll_1w[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above 20-week high + volume spike
            if (close[i] > hh_ll_1w[i] and 
                volume[i] > vol_ma[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # Short: Break below 20-week low + volume spike
            elif (close[i] < ll_ll_1w[i] and 
                  volume[i] > vol_ma[i] * 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close below 20-week low or volume drops below average
            if (close[i] < ll_ll_1w[i] or 
                volume[i] < vol_ma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above 20-week high or volume drops below average
            if (close[i] > hh_ll_1w[i] or 
                volume[i] < vol_ma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Daily price action using 20-week high/low breakouts with volume confirmation.
# In bull markets, price breaks above weekly resistance and continues upward.
# In bear markets, price breaks below weekly support and continues downward.
# Volume confirmation ensures institutional participation, reducing false breakouts.
# Position size 0.25 limits drawdown during adverse markets (e.g., 2022 BTC crash).
# Weekly timeframe provides structural context, reducing whipsaws in daily noise.
# Target: 10-25 trades/year to minimize fee drag while capturing major trends.