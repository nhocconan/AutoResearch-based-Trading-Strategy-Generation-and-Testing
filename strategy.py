#!/usr/bin/env python3
# Hypothesis: 1-day Donchian(20) breakout with 1-week EMA50 trend filter and volume confirmation.
# Long when price breaks above the 20-day high with 1-week EMA50 uptrend and volume > 1.5x average.
# Short when price breaks below the 20-day low with 1-week EMA50 downtrend and volume > 1.5x average.
# Exit when price crosses the 10-day EMA in the opposite direction or after a fixed 5-day hold.
# This strategy targets major trend breaks with low frequency to minimize fee drag, suitable for 1d timeframe.
# The 1-week EMA filter ensures alignment with the higher timeframe trend, reducing false breakouts.
# Volume confirmation adds conviction to breakouts.
# Expected trades: ~10-25 per year (40-100 over 4 years), within target range.

name = "1d_Donchian20_1wEMA50_VolumeConfirm"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need enough data for Donchian and EMA
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1-week EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate 10-day EMA for exit
    ema10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0  # For time-based exit
    
    start_idx = 50  # Need enough data for Donchian and EMA calculations
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema50_1w_aligned[i]) or np.isnan(ema10[i]) or np.isnan(vol_confirm[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        if position == 0:
            bars_since_entry = 0
            # Calculate 20-day Donchian channels
            highest_high = np.max(high[i-19:i+1]) if i >= 19 else np.nan
            lowest_low = np.min(low[i-19:i+1]) if i >= 19 else np.nan
            
            if np.isnan(highest_high) or np.isnan(lowest_low):
                continue
            
            # Enter long: price breaks above 20-day high, 1-week EMA50 uptrend, volume confirmation
            if (close[i] > highest_high and 
                ema50_1w_aligned[i] > ema50_1w_aligned[i-1] and  # EMA rising
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below 20-day low, 1-week EMA50 downtrend, volume confirmation
            elif (close[i] < lowest_low and 
                  ema50_1w_aligned[i] < ema50_1w_aligned[i-1] and  # EMA falling
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            bars_since_entry += 1
            # Exit conditions: 
            # 1. Price crosses 10-day EMA in opposite direction
            # 2. Maximum hold of 5 days (to prevent excessive whipsaw in choppy markets)
            if position == 1:  # Long position
                if close[i] < ema10[i] or bars_since_entry >= 5:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            elif position == -1:  # Short position
                if close[i] > ema10[i] or bars_since_entry >= 5:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals