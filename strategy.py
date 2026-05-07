#!/usr/bin/env python3
# 4h_Donchian20_Breakout_1dTrend_Filter_Volume
# Hypothesis: 4h strategy using Donchian(20) breakout with 1d EMA50 trend filter and volume spike.
# Long when price breaks above Donchian(20) high, price above 1d EMA50 (uptrend), and volume > 1.5x average.
# Short when price breaks below Donchian(20) low, price below 1d EMA50 (downtrend), and volume > 1.5x average.
# Exits when price returns to the opposite Donchian level (e.g., long exits at Donchian low).
# Designed to capture trends in both bull and bear markets with strict entry conditions to avoid overtrading.
# Uses tight exits and volume confirmation to improve win rate and reduce false breakouts.

name = "4h_Donchian20_Breakout_1dTrend_Filter_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate EMA50 on 1d close for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Donchian(20) channels on 4h data
    # Use rolling window with min_periods to avoid look-ahead
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume spike detection: 1.5x average volume (20-period for stability)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Ensure we have Donchian and EMA50 data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high, price above 1d EMA50 (uptrend), volume spike (>1.5x)
            if (close[i] > donchian_high[i] and 
                close[i] > ema50_1d_aligned[i] and 
                volume[i] > 1.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low, price below 1d EMA50 (downtrend), volume spike (>1.5x)
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema50_1d_aligned[i] and 
                  volume[i] > 1.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns to or below Donchian low (opposite level)
            if close[i] <= donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to or above Donchian high (opposite level)
            if close[i] >= donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals