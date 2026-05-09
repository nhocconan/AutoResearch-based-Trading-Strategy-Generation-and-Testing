#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ehlers Fisher Transform on price with 1d trend filter and volume spike.
# Fisher Transform identifies extreme price reversals with high precision.
# Long when Fisher crosses above -1.5 in 1d uptrend with volume spike.
# Short when Fisher crosses below +1.5 in 1d downtrend with volume spike.
# Works in both bull (follow 1d uptrend) and bear (follow 1d downtrend) by catching reversals.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.
name = "6h_EhlersFisher_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and Fisher Transform calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 2.0x 20-period EMA (high threshold for fewer trades)
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (2.0 * vol_ema20)
    
    # Ehlers Fisher Transform on 6h prices (length=10)
    # Fisher Transform = 0.5 * ln((1+X)/(1-X)) where X is normalized price
    def fish(price_array, length=10):
        if len(price_array) < length:
            return np.full_like(price_array, np.nan)
        highest = np.max(price_array[-length:])
        lowest = np.min(price_array[-length:])
        range_val = highest - lowest
        if range_val == 0:
            return 0.0
        # Avoid division by zero and clamp X to [-0.999, 0.999]
        X = 2 * (price_array[-1] - lowest) / range_val - 1
        X = max(min(X, 0.999), -0.999)
        if X == 1 or X == -1:
            X = 0.999 * np.sign(X) if X != 0 else 0
        fish_val = 0.5 * np.log((1 + X) / (1 - X))
        # Recursive smoothing: fish[t] = 0.5 * fish_val[t] + 0.5 * fish[t-1]
        return fish_val  # We'll handle smoothing in loop
    
    # Precompute Fisher values for all points
    fish_raw = np.full(n, np.nan)
    for i in range(9, n):  # Need 10 bars for length=10
        highest = np.max(high[i-9:i+1])
        lowest = np.min(low[i-9:i+1])
        range_val = highest - lowest
        if range_val == 0:
            fish_raw[i] = 0.0
        else:
            X = 2 * (close[i] - lowest) / range_val - 1
            X = max(min(X, 0.999), -0.999)
            if abs(X) >= 1.0:
                X = 0.999 * np.sign(X)
            fish_raw[i] = 0.5 * np.log((1 + X) / (1 - X))
    
    # Apply IIR smoothing: fish[t] = 0.5 * fish_raw[t] + 0.5 * fish[t-1]
    fish = np.full(n, np.nan)
    for i in range(n):
        if np.isnan(fish_raw[i]):
            fish[i] = np.nan
        elif i == 0:
            fish[i] = fish_raw[i]
        else:
            fish[i] = 0.5 * fish_raw[i] + 0.5 * fish[i-1]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 10  # Need enough data for Fisher Transform
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(fish[i]) or 
            np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Enter long: Fisher crosses above -1.5 + 1d uptrend + volume spike
            if (fish[i] > -1.5 and fish[i-1] <= -1.5 and 
                price > ema_34_1d_aligned[i] and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Fisher crosses below +1.5 + 1d downtrend + volume spike
            elif (fish[i] < 1.5 and fish[i-1] >= 1.5 and 
                  price < ema_34_1d_aligned[i] and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Fisher crosses below -1.5 or trend reverses
            if fish[i] < -1.5 or price < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Fisher crosses above +1.5 or trend reverses
            if fish[i] > 1.5 or price > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals