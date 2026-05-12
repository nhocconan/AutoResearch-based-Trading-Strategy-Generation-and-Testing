#!/usr/bin/env python3
# 6H_EHLERS_FISHER_TRANSFORM_1D_TREND_FILTER
# Hypothesis: Ehlers Fisher Transform identifies turning points in price cycles with minimal lag.
# Combined with 1D trend filter to avoid counter-trend trades. Works in both bull and bear markets:
# - In bull markets: captures pullback reversals within uptrend
# - In bear markets: captures bounces within downtrend
# Target: 20-30 trades/year on 6h timeframe.

name = "6H_EHLERS_FISHER_TRANSFORM_1D_TREND_FILTER"
timeframe = "6h"
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
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # EMA50 for trend filter
    ema50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50)
    
    # Ehlers Fisher Transform (9-period)
    price = (high + low) / 2
    # Normalize price to 0-1 range over 9 periods
    max_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    min_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    # Avoid division by zero
    price_range = max_high - min_low
    price_range = np.where(price_range == 0, 1, price_range)
    value1 = 2 * ((price - min_low) / price_range - 0.5)
    # Smooth value1
    value2 = np.zeros_like(value1)
    for i in range(9, n):
        if i == 9:
            value2[i] = value1[i]
        else:
            value2[i] = 0.33 * value1[i] + 0.67 * value2[i-1]
    # Clamp to [-0.999, 0.999] to avoid math domain error
    value2 = np.clip(value2, -0.999, 0.999)
    # Fisher Transform
    fish = 0.5 * np.log((1 + value2) / (1 - value2))
    # Smooth Fisher
    fish_smooth = np.zeros_like(fish)
    for i in range(9, n):
        if i == 9:
            fish_smooth[i] = fish[i]
        else:
            fish_smooth[i] = 0.5 * fish[i] + 0.5 * fish_smooth[i-1]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if np.isnan(ema50_aligned[i]) or np.isnan(fish_smooth[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Fisher crosses above -1.5 in uptrend
            if (fish_smooth[i] > -1.5 and fish_smooth[i-1] <= -1.5 and 
                close[i] > ema50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Fisher crosses below +1.5 in downtrend
            elif (fish_smooth[i] < 1.5 and fish_smooth[i-1] >= 1.5 and 
                  close[i] < ema50_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Fisher crosses below -1.5 or trend reversal
            if (fish_smooth[i] < -1.5 and fish_smooth[i-1] >= -1.5) or \
               (close[i] <= ema50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Fisher crosses above +1.5 or trend reversal
            if (fish_smooth[i] > 1.5 and fish_smooth[i-1] <= 1.5) or \
               (close[i] >= ema50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals