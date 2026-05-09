#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Fisher Transform (Ehlers) with 1d trend filter and volume confirmation.
# Fisher Transform normalizes price to create clear turning points.
# Long when Fisher crosses above -1.5 with 1d uptrend and volume confirmation.
# Short when Fisher crosses below +1.5 with 1d downtrend and volume confirmation.
# Designed to catch reversals in both bull and bear markets with clear signals.
# Works well in ranging markets (2025-2026) and catches trend changes.
name = "6h_FisherTransform_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Fisher Transform (Ehlers)
    def fishert_transform(high_arr, low_arr, length=10):
        # Median price
        hl2 = (high_arr + low_arr) / 2
        
        # Min and max over lookback period
        min_hll = np.full_like(hl2, np.nan)
        max_hll = np.full_like(hl2, np.nan)
        
        for i in range(length-1, len(hl2)):
            min_hll[i] = np.min(hl2[i-length+1:i+1])
            max_hll[i] = np.max(hl2[i-length+1:i+1])
        
        # Normalize to [-1, 1] range
        numerator = 2 * ((hl2 - min_hll) / (max_hll - min_hll + 1e-10) - 0.5)
        # Smooth with 2-period EMA
        smoothed = np.full_like(numerator, np.nan)
        ema_weight = 2 / (2 + 1)  # 2-period EMA
        for i in range(len(numerator)):
            if np.isnan(numerator[i]):
                smoothed[i] = np.nan
            elif np.isnan(smoothed[i-1]) if i > 0 else False:
                smoothed[i] = numerator[i]
            else:
                smoothed[i] = ema_weight * numerator[i] + (1 - ema_weight) * smoothed[i-1]
        
        # Fisher Transform: 0.5 * ln((1 + smoothed) / (1 - smoothed))
        fish = np.full_like(smoothed, np.nan)
        for i in range(len(smoothed)):
            if np.isnan(smoothed[i]) or abs(smoothed[i]) >= 1:
                fish[i] = np.nan
            else:
                fish[i] = 0.5 * np.log((1 + smoothed[i]) / (1 - smoothed[i]))
        
        return fish
    
    fish = fishert_transform(high, low, length=10)
    
    # 1d EMA trend filter
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(fish[i]) or np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: Fisher crosses above -1.5 (from below) + 1d uptrend + volume
            if (fish[i] > -1.5 and fish[i-1] <= -1.5 and 
                price > ema_1d_aligned[i] and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Fisher crosses below +1.5 (from above) + 1d downtrend + volume
            elif (fish[i] < 1.5 and fish[i-1] >= 1.5 and 
                  price < ema_1d_aligned[i] and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Fisher crosses below +1.5 or 1d trend turns down
            if fish[i] < 1.5 or price < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Fisher crosses above -1.5 or 1d trend turns up
            if fish[i] > -1.5 or price > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals