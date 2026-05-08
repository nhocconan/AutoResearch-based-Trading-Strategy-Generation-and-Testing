#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Turtle Trading System (Donchian Breakout) with 4h trend filter and volume confirmation
# Long when price breaks above 20-period high + 4h close > 4h EMA50 + volume spike
# Short when price breaks below 20-period low + 4h close < 4h EMA50 + volume spike
# Designed to capture trends in both bull and bear markets with low trade frequency.
# Target: 60-150 total trades over 4 years = 15-37/year for 1h.

name = "1h_TurtleBreakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data once
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA(50) for trend direction
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 20-period Donchian channels
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(high_max[i]) or 
            np.isnan(low_min[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_4h_val = ema50_4h_aligned[i]
        donchian_high = high_max[i]
        donchian_low = low_min[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price breaks above 20-period high + 4h uptrend + volume spike
            if (close[i] > donchian_high and 
                close[i] > ema50_4h_val and 
                vol_spike):
                signals[i] = 0.20
                position = 1
            # Enter short: price breaks below 20-period low + 4h downtrend + volume spike
            elif (close[i] < donchian_low and 
                  close[i] < ema50_4h_val and 
                  vol_spike):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price breaks below 20-period low
            if close[i] < donchian_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price breaks above 20-period high
            if close[i] > donchian_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals