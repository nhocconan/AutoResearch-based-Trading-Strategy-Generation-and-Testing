#!/usr/bin/env python3
# 6H_FISHER_TRANSFORM_1D_TREND_FILTER
# Hypothesis: Ehlers Fisher Transform on 1d prices identifies extreme turning points.
# In 1d uptrend (EMA34), go long when Fisher crosses above -1.5; in downtrend, go short when crosses below +1.5.
# Works in both bull and bear markets: trend filter avoids counter-trend trades, Fisher captures reversals within trend.
# Target: 15-25 trades/year on 6h timeframe.

name = "6H_FISHER_TRANSFORM_1D_TREND_FILTER"
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
    
    # Daily data for Fisher Transform and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Fisher Transform (Ehlers) on daily close
    # Price normalized to [-1, 1] over lookback period
    high = df_1d['high'].values
    low = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Normalize price to [-1, 1] range over 10-period lookback
    def normalize_price(high, low, close, length):
        highest_high = np.maximum.accumulate(high)
        lowest_low = np.minimum.accumulate(low)
        # Avoid division by zero
        range_val = highest_high - lowest_low
        range_val = np.where(range_val == 0, 1, range_val)
        value = 2 * ((close - lowest_low) / range_val) - 1
        # Clamp to [-0.999, 0.999] to avoid infinity in Fisher
        value = np.clip(value, -0.999, 0.999)
        return value
    
    price_norm = normalize_price(high, low, close_1d, 10)
    
    # Fisher Transform: 0.5 * ln((1+value)/(1-value))
    fish = 0.5 * np.log((1 + price_norm) / (1 - price_norm))
    # Smoothed Fisher (signal line)
    fish_smooth = np.zeros_like(fish)
    alpha = 0.5
    for i in range(1, len(fish)):
        fish_smooth[i] = alpha * fish[i] + (1 - alpha) * fish_smooth[i-1]
        # Prevent extreme values
        fish_smooth[i] = np.clip(fish_smooth[i], -5, 5)
    
    # EMA34 for trend filter
    ema34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align to 6h timeframe
    fish_smooth_aligned = align_htf_to_ltf(prices, df_1d, fish_smooth)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(fish_smooth_aligned[i]) or np.isnan(ema34_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: 1d uptrend + Fisher crosses above -1.5 (from below)
            if (close[i] > ema34_aligned[i] and 
                fish_smooth_aligned[i] > -1.5 and 
                fish_smooth_aligned[i-1] <= -1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: 1d downtrend + Fisher crosses below +1.5 (from above)
            elif (close[i] < ema34_aligned[i] and 
                  fish_smooth_aligned[i] < 1.5 and 
                  fish_smooth_aligned[i-1] >= 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Trend reversal or Fisher crosses below -1.5
            if (close[i] <= ema34_aligned[i] or 
                fish_smooth_aligned[i] < -1.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend reversal or Fisher crosses above +1.5
            if (close[i] >= ema34_aligned[i] or 
                fish_smooth_aligned[i] > 1.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals