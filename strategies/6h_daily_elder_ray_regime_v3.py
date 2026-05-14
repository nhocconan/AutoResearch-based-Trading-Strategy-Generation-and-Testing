#!/usr/bin/env python3
# 6h_daily_elder_ray_regime_v3
# Hypothesis: Elder Ray (Bull Power/Bear Power) with 1d regime filter on 6h timeframe.
# Long: Bull Power > 0 AND Bear Power < 0 AND close > 1d EMA50 (bull regime)
# Short: Bear Power < 0 AND Bull Power > 0 AND close < 1d EMA50 (bear regime)
# Exit: Opposite signal or Elder Power divergence (Bull Power makes lower high while price makes higher high for longs, vice versa for shorts)
# Uses 6h primary timeframe with 1d HTF for EMA50 regime filter.
# Target: 50-150 total trades over 4 years to minimize fee drag and avoid overtrading.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_daily_elder_ray_regime_v3"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate EMA13 for Elder Ray power with min_periods
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high - ema13
    # Bear Power = Low - EMA13
    bear_power = low - ema13
    
    # Get 1d data for regime filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate EMA50 on 1d with min_periods
    close_1d_s = pd.Series(close_1d)
    ema50_1d = close_1d_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Track previous Bull Power and Bear Power for divergence detection
    prev_bull_power = bull_power[0] if not np.isnan(bull_power[0]) else 0
    prev_bear_power = bear_power[0] if not np.isnan(bear_power[0]) else 0
    
    for i in range(13, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(close[i])):
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
            # Update previous values for next iteration
            if not np.isnan(bull_power[i]):
                prev_bull_power = bull_power[i]
            if not np.isnan(bear_power[i]):
                prev_bear_power = bear_power[i]
            continue
        
        if position == 1:  # Long position
            # Exit conditions:
            # 1. Opposite signal (Bear Power > 0 AND Bull Power < 0) - regime change
            # 2. Bearish divergence: Bull Power makes lower high while price makes higher high
            bull_divergence = (bull_power[i] < prev_bull_power and 
                              close[i] > close[i-1] and 
                              prev_bull_power > 0)  # Only when previously bullish
            
            if (bear_power[i] > 0 and bull_power[i] < 0) or bull_divergence:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions:
            # 1. Opposite signal (Bull Power > 0 AND Bear Power < 0) - regime change
            # 2. Bullish divergence: Bear Power makes higher low while price makes lower low
            bear_divergence = (bear_power[i] > prev_bear_power and 
                              close[i] < close[i-1] and 
                              prev_bear_power < 0)  # Only when previously bearish
            
            if (bull_power[i] > 0 and bear_power[i] < 0) or bear_divergence:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Bull Power > 0 AND Bear Power < 0 AND close > 1d EMA50 (bull regime)
            if bull_power[i] > 0 and bear_power[i] < 0 and close[i] > ema50_1d_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: Bear Power < 0 AND Bull Power > 0 AND close < 1d EMA50 (bear regime)
            elif bear_power[i] < 0 and bull_power[i] > 0 and close[i] < ema50_1d_aligned[i]:
                position = -1
                signals[i] = -0.25
        
        # Update previous values for next iteration
        prev_bull_power = bull_power[i]
        prev_bear_power = bear_power[i]
    
    return signals