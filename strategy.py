#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian Breakout with 1d Volume Confirmation and Volatility Filter
# Uses Donchian channel (20-period high/low) to identify breakouts in the direction of 1d trend.
# Long when price breaks above 20-period high and 1d EMA50 > EMA200 (bullish).
# Short when price breaks below 20-period low and 1d EMA50 < EMA200 (bearish).
# Volume confirmation requires > 1.5x 20-bar median volume to avoid false breakouts.
# Designed to work in bull markets (breakouts with momentum) and bear markets (breakdowns with follow-through).
# Conservative sizing (0.25) to limit trade frequency and avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day EMA for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean()
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean()
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d.values)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d.values)
    
    # Donchian channel (20-period) on 4h
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min()
    
    # Volume confirmation: current > 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(vol_threshold[i])):
            continue
        
        # Long: price breaks above 20-period high, 1d bullish trend, volume spike
        if (close[i] > high_max[i] and 
            ema50_1d_aligned[i] > ema200_1d_aligned[i] and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short: price breaks below 20-period low, 1d bearish trend, volume spike
        elif (close[i] < low_min[i] and 
              ema50_1d_aligned[i] < ema200_1d_aligned[i] and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit: price returns to middle of Donchian channel or trend changes
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < (high_max[i] + low_min[i]) / 2) or
               (signals[i-1] == -0.25 and close[i] > (high_max[i] + low_min[i]) / 2))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_Donchian_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0