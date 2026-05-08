#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h EMA50 trend filter + volume confirmation
# Donchian breakouts capture momentum in trending markets, while EMA50 filters for trend direction.
# Volume confirmation ensures breakouts are supported by participation. Stops via signal=0 on adverse moves.
# Targets 20-40 trades per year (~80-160 total over 4 years) to minimize fee drag.

name = "4h_Donchian20_12hEMA50_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels: 20-period high and low
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate EMA50 on 12h close for trend filter
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_slope = ema50_12h[1:] - ema50_12h[:-1]  # slope: positive = uptrend
    ema50_12h_slope = np.concatenate([[0], ema50_12h_slope])  # align length
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    ema50_12h_slope_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h_slope)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_conf = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for Donchian and EMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(ema50_12h_slope_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        high_20_val = high_20[i]
        low_20_val = low_20[i]
        ema50_val = ema50_12h_aligned[i]
        ema50_slope = ema50_12h_slope_aligned[i]
        vol_conf_val = vol_conf[i]
        
        if position == 0:
            # Enter long: close breaks above Donchian high, EMA50 uptrend, volume confirmation
            if close[i] > high_20_val and ema50_slope > 0 and vol_conf_val:
                signals[i] = 0.25
                position = 1
            # Enter short: close breaks below Donchian low, EMA50 downtrend, volume confirmation
            elif close[i] < low_20_val and ema50_slope < 0 and vol_conf_val:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: close breaks below Donchian low or EMA50 turns down
            if close[i] < low_20_val or ema50_slope < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: close breaks above Donchian high or EMA50 turns up
            if close[i] > high_20_val or ema50_slope > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals