#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h trend filter and volume spike confirmation.
# Donchian breakouts capture momentum bursts. 12h EMA50 filters direction to avoid counter-trend trades.
# Volume spike (>2x 20-period average) confirms institutional participation and reduces false breakouts.
# Designed for 6-12 trades/year per symbol (24-48 total over 4 years) to minimize fee drag.
# Works in bull markets via breakout continuation and in bear markets via short breakdowns.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # 50-period EMA on 12h close for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume filter: volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long breakout: price above upper band in uptrend with volume
        if close[i] > high_max[i] and close[i] > ema50_12h_aligned[i] and volume_filter[i]:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short breakdown: price below lower band in downtrend with volume
        elif close[i] < low_min[i] and close[i] < ema50_12h_aligned[i] and volume_filter[i]:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: reverse signal or loss of trend/volume
        else:
            # Exit long if price crosses below lower band or trend fails
            if position == 1 and (close[i] < low_min[i] or close[i] < ema50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            # Exit short if price crosses above upper band or trend fails
            elif position == -1 and (close[i] > high_max[i] or close[i] > ema50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            # Hold current position if still valid
            elif position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_Donchian20_12hEMA50_VolumeFilter"
timeframe = "6h"
leverage = 1.0