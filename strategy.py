#!/usr/bin/env python3
# 6h_12h_Donchian20_Breakout_1dTrend_Volume
# Hypothesis: On 6h timeframe, trade breakouts from 12h Donchian(20) channels when aligned with 1d EMA50 trend and volume confirmation.
# Works in bull markets via long breakouts above upper band with uptrend; works in bear markets via short breakdowns below lower band with downtrend.
# Volume > 1.5x 20-period average confirms breakout strength. Targets 15-25 trades per year per symbol.

name = "6h_12h_Donchian20_Breakout_1dTrend_Volume"
timeframe = "6h"
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
    
    # Get 12h data ONCE before loop for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h Donchian channels (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Rolling max/min for Donchian channels
    high_series = pd.Series(high_12h)
    low_series = pd.Series(low_12h)
    upper_12h = high_series.rolling(window=20, min_periods=20).max().values
    lower_12h = low_series.rolling(window=20, min_periods=20).min().values
    
    # Get 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h and 1d indicators to 6h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_12h, upper_12h)
    lower_aligned = align_htf_to_ltf(prices, df_12h, lower_12h)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume average for confirmation (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above upper Donchian, volume spike, and price above 1d EMA50 (uptrend)
            if (close[i] > upper_aligned[i] and 
                volume[i] > 1.5 * volume_ma[i] and
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below lower Donchian, volume spike, and price below 1d EMA50 (downtrend)
            elif (close[i] < lower_aligned[i] and 
                  volume[i] > 1.5 * volume_ma[i] and
                  close[i] < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price below lower Donchian or trend reversal (below EMA50)
            if close[i] < lower_aligned[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above upper Donchian or trend reversal (above EMA50)
            if close[i] > upper_aligned[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals