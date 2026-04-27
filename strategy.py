#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation.
# Long when price breaks above 20-period high with bullish weekly trend and volume spike.
# Short when price breaks below 20-period low with bearish weekly trend and volume spike.
# Exit when price returns to 20-period midpoint (mean reversion).
# Uses weekly timeframe for trend filter to reduce noise and improve win rate in both bull and bear markets.
# Target: 12-37 trades/year (50-150 over 4 years) to minimize fee drag while capturing strong moves.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 2:
        return np.zeros(n)
    
    close_w = df_w['close'].values
    
    # Calculate weekly EMA20 for trend filter
    ema20_w = pd.Series(close_w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_w_aligned = align_htf_to_ltf(prices, df_w, ema20_w)
    
    # Calculate Donchian channels (20-period high/low and midpoint)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_max + low_min) / 2
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema20_w_aligned[i]) or np.isnan(high_max[i]) or 
            np.isnan(low_min[i]) or np.isnan(donchian_mid[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: price breaks above 20-period high, above weekly EMA20, volume spike
        if (close[i] > high_max[i] and 
            close[i] > ema20_w_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short condition: price breaks below 20-period low, below weekly EMA20, volume spike
        elif (close[i] < low_min[i] and 
              close[i] < ema20_w_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: price returns to Donchian midpoint (mean reversion)
        elif position == 1 and close[i] < donchian_mid[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > donchian_mid[i]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_Donchian20_1wEMA20_VolumeFilter"
timeframe = "6h"
leverage = 1.0