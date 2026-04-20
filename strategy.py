#!/usr/bin/env python3
# 1d_1w_Donchian_Breakout_Volume_Trend
# Hypothesis: On 1d timeframe, trade Donchian(20) breakouts with volume confirmation and 1w EMA trend filter.
# In trending markets (price > 1w EMA), take long breakouts above 20-day high; in ranging markets (price < 1w EMA),
# take short breakdowns below 20-day low. Uses volume > 1.5x 20-day average for confirmation.
# Targets 10-25 trades/year by requiring confluence of breakout, volume, and trend filter.

name = "1d_1w_Donchian_Breakout_Volume_Trend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Calculate 1w EMA(21) for trend filter
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Calculate Donchian channels (20-period) on 1d
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(ema_21_1w_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Trend filter: price > 1w EMA = uptrend, price < 1w EMA = downtrend
            if close[i] > ema_21_1w_aligned[i]:
                # Uptrend: look for long breakout above 20-day high
                if (close[i] > high_max[i] and 
                    volume[i] > 1.5 * volume_ma[i]):
                    signals[i] = 0.25
                    position = 1
            else:
                # Downtrend: look for short breakdown below 20-day low
                if (close[i] < low_min[i] and 
                    volume[i] > 1.5 * volume_ma[i]):
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: price falls below 20-day low or trend reverses
            if close[i] < low_min[i] or close[i] < ema_21_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above 20-day high or trend reverses
            if close[i] > high_max[i] or close[i] > ema_21_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals