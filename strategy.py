#!/usr/bin/env python3
# 6h_Aroon_AroonOsc_TrendStrength_With_1dTrend
# Hypothesis: Aroon oscillator identifies strong trends with low lag, and when combined with
# 1-day trend filter and volume confirmation, it captures continuation moves in both bull and bear markets.
# Aroon > 70 indicates strong uptrend, Aroon < -70 indicates strong downtrend.
# We enter on Aroon cross above/below zero with 1d trend alignment and volume spike.
# Exit when Aroon crosses back below/above zero or trend breaks.
# This avoids whipsaws in ranging markets while catching strong trends.

name = "6h_Aroon_AroonOsc_TrendStrength_With_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Aroon oscillator (25-period) on 6h chart
    # Aroon Up = ((25 - periods since 25-period high) / 25) * 100
    # Aroon Down = ((25 - periods since 25-period low) / 25) * 100
    # Aroon Oscillator = Aroon Up - Aroon Down
    lookback = 25
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    # Calculate rolling index of max/min
    def rolling_argmax(arr, window):
        return pd.Series(arr).rolling(window, min_periods=1).apply(lambda x: np.argmax(x), raw=True)
    def rolling_argmin(arr, window):
        return pd.Series(arr).rolling(window, min_periods=1).apply(lambda x: np.argmin(x), raw=True)
    
    # Since we can't use apply easily in vectorized way, we'll calculate manually in loop later
    # Instead, we'll compute Aroon using standard formulas with min_periods
    high_rolling_max = high_series.rolling(window=lookback, min_periods=lookback).max()
    low_rolling_min = low_series.rolling(window=lookback, min_periods=lookback).min()
    
    # Calculate periods since high/low
    # We'll use expanding window approach for efficiency
    periods_since_high = np.zeros_like(high)
    periods_since_low = np.zeros_like(low)
    
    # Initialize
    max_idx = 0
    min_idx = 0
    for i in range(n):
        if i < lookback:
            periods_since_high[i] = i
            periods_since_low[i] = i
        else:
            # Update max/min index
            if high[i] >= high[max_idx]:
                max_idx = i
            if low[i] <= low[min_idx]:
                min_idx = i
            
            # Reset if outside lookback window
            if max_idx < i - lookback + 1:
                # Find new max in window
                window_start = i - lookback + 1
                max_idx = window_start + np.argmax(high[window_start:i+1])
            if min_idx < i - lookback + 1:
                # Find new min in window
                window_start = i - lookback + 1
                min_idx = window_start + np.argmin(low[window_start:i+1])
            
            periods_since_high[i] = i - max_idx
            periods_since_low[i] = i - min_idx
    
    aroon_up = ((lookback - periods_since_high) / lookback) * 100
    aroon_down = ((lookback - periods_since_low) / lookback) * 100
    aroon_osc = aroon_up - aroon_down  # -100 to +100
    
    # Calculate daily EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation (20-period MA on 6h chart)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Aroon (25), EMA50_1d (50), volume MA (20)
    start_idx = max(25, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(aroon_osc[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Daily trend filter
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        # Aroon signals: zero cross
        if i > 0:
            cross_above_zero = (aroon_osc[i] > 0) and (aroon_osc[i-1] <= 0)
            cross_below_zero = (aroon_osc[i] < 0) and (aroon_osc[i-1] >= 0)
        else:
            cross_above_zero = False
            cross_below_zero = False
        
        if position == 0:
            # Long entry: strong uptrend (Aroon > 0) + 1d uptrend + volume
            if aroon_osc[i] > 0 and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: strong downtrend (Aroon < 0) + 1d downtrend + volume
            elif aroon_osc[i] < 0 and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend weakness or reversal
            if aroon_osc[i] <= 0 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend weakness or reversal
            if aroon_osc[i] >= 0 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals