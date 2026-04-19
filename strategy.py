#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d timeframe with 1w trend alignment using 1w EMA20 and 1d volume spike.
# Enters long when price > 1w EMA20 and volume > 2x 20-day average, exits when price < 1w EMA20.
# Shorts when price < 1w EMA20 and volume > 2x average, exits when price > 1w EMA20.
# Uses weekly trend to avoid whipsaw in both bull and bear markets.
# Volume spike confirms institutional interest. Targets 10-25 trades/year.
name = "1d_1w_EMA20_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Get 1w data for EMA20 trend (called ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume filter: volume > 2.0 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_20_1w_aligned[i]) or np.isnan(volume_ma[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above 1w EMA20 AND volume spike
            if close[i] > ema_20_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below 1w EMA20 AND volume spike
            elif close[i] < ema_20_1w_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price crosses below 1w EMA20
            if close[i] < ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price crosses above 1w EMA20
            if close[i] > ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals