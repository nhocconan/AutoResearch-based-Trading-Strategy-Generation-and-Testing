#!/usr/bin/env python3
# 6H_EHLERS_FISHER_TRANSFORM_1D_TREND
# Hypothesis: Ehlers Fisher Transform on 1d identifies turning points in cyclical markets.
# When Fisher crosses above -1.5, it signals the end of a sell cycle and potential uptrend.
# When Fisher crosses below +1.5, it signals the end of a buy cycle and potential downtrend.
# Combined with 12h trend filter (EMA50) and volume confirmation to avoid false signals.
# Works in both bull and bear markets by catching reversals at extremes.
# Target: 15-35 trades/year on 6h timeframe to avoid overtrading.

name = "6H_EHLERS_FISHER_TRANSFORM_1D_TREND"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Fisher Transform
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 15:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ehlers Fisher Transform (price normalization to [-1, 1])
    def fishert_transform(price, length=10):
        # Normalize price to [-1, 1] over the lookback period
        highest = np.maximum.accumulate(price)
        lowest = np.minimum.accumulate(price)
        # Avoid division by zero
        diff = highest - lowest
        diff = np.where(diff == 0, 1e-10, diff)
        value = 2 * ((price - lowest) / diff) - 1
        # Smooth with a small smoothing factor
        value = np.where((value > 0.999), 0.999, value)
        value = np.where((value < -0.999), -0.999, value)
        # Fisher transform
        fish = 0.5 * np.log((1 + value) / (1 - value) + 1e-10)
        # Apply exponential moving average for smoothing
        fish = pd.Series(fish).ewm(alpha=0.2, adjust=False).fillna(0).values
        return fish
    
    # Apply Fisher Transform to typical price
    typical_price_1d = (high_1d + low_1d + close_1d) / 3
    fish = fishert_transform(typical_price_1d, 10)
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # EMA50 for 12h trend
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume spike: current 6h volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    volume_spike = volume > 2.0 * vol_ma
    
    # Align all data to 6h timeframe
    fish_aligned = align_htf_to_ltf(prices, df_1d, fish)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if any critical data is not ready
        if (np.isnan(fish_aligned[i]) or np.isnan(ema50_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Fisher crosses above -1.5 (end of sell cycle) with volume in uptrend
            if (fish_aligned[i-1] <= -1.5 and fish_aligned[i] > -1.5 and 
                volume_spike[i] and 
                close[i] > ema50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Fisher crosses below +1.5 (end of buy cycle) with volume in downtrend
            elif (fish_aligned[i-1] >= 1.5 and fish_aligned[i] < 1.5 and 
                  volume_spike[i] and 
                  close[i] < ema50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Fisher crosses below +1.5 (sell cycle begins) or trend reversal
            if (fish_aligned[i] < 1.5 and fish_aligned[i-1] >= 1.5) or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Fisher crosses above -1.5 (buy cycle begins) or trend reversal
            if (fish_aligned[i] > -1.5 and fish_aligned[i-1] <= -1.5) or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals