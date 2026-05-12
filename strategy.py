#!/usr/bin/env python3
# 6H_EHLERS_FISHER_TRANSFORM_1D_TREND_FILTER
# Hypothesis: Ehlers Fisher Transform identifies extreme price reversals in oscillator form.
# On 6h timeframe, we use 1D trend filter to avoid counter-trend trades.
# Fisher crosses above -1.5 signal potential long reversals from oversold.
# Fisher crosses below +1.5 signal potential short reversals from overbought.
# Works in both bull and bear markets by catching reversals at extremes.
# Target: 15-25 trades/year on 6h timeframe.

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
    
    # Daily data for trend filter and Fisher Transform
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily median price for Fisher Transform
    hl2 = (df_1d['high'].values + df_1d['low'].values) / 2
    
    # Ehlers Fisher Transform (9-period)
    # Step 1: Normalize price to -1 to +1 range over lookback period
    length = 9
    highest_high = np.full(len(hl2), np.nan)
    lowest_low = np.full(len(hl2), np.nan)
    
    for i in range(length-1, len(hl2)):
        highest_high[i] = np.max(hl2[i-length+1:i+1])
        lowest_low[i] = np.min(hl2[i-length+1:i+1])
    
    # Avoid division by zero
    diff = highest_high - lowest_low
    diff[diff == 0] = 1e-10
    
    # Normalized price
    normalized = np.where(np.isnan(highest_high), np.nan, 
                         2 * ((hl2 - lowest_low) / diff) - 1)
    
    # Smooth normalized price (optional but recommended)
    smoothed = np.full(len(normalized), np.nan)
    alpha = 0.5  # Smoothing factor
    for i in range(len(normalized)):
        if np.isnan(normalized[i]):
            smoothed[i] = np.nan
        elif i == 0:
            smoothed[i] = normalized[i]
        else:
            smoothed[i] = alpha * normalized[i] + (1 - alpha) * smoothed[i-1]
    
    # Fisher Transform
    # Clamp smoothed values to avoid domain error in log
    smoothed_clamped = np.clip(smoothed, -0.999, 0.999)
    fisher = np.where(np.isnan(smoothed_clamped), np.nan,
                     0.5 * np.log((1 + smoothed_clamped) / (1 - smoothed_clamped)))
    
    # EMA34 for trend filter
    ema34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align to 6h timeframe
    fisher_aligned = align_htf_to_ltf(prices, df_1d, fisher)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(fisher_aligned[i]) or np.isnan(ema34_aligned[i]) or 
            i == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Fisher crosses above -1.5 (coming from oversold) in uptrend
            if (fisher_aligned[i] > -1.5 and fisher_aligned[i-1] <= -1.5 and 
                close[i] > ema34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Fisher crosses below +1.5 (coming from overbought) in downtrend
            elif (fisher_aligned[i] < 1.5 and fisher_aligned[i-1] >= 1.5 and 
                  close[i] < ema34_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Fisher crosses below +1.5 or trend reversal
            if (fisher_aligned[i] < 1.5 and fisher_aligned[i-1] >= 1.5) or \
               close[i] <= ema34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Fisher crosses above -1.5 or trend reversal
            if (fisher_aligned[i] > -1.5 and fisher_aligned[i-1] <= -1.5) or \
               close[i] >= ema34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals