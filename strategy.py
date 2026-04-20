#!/usr/bin/env python3
# 1d_Donchian_Breakout_Volume_TrendFilter
# Hypothesis: Daily Donchian(20) breakouts capture strong trends. Volume confirmation filters false breakouts.
# Trend filter: price above/below 50-day EMA ensures alignment with medium-term trend.
# Works in bull markets via upward breakouts, in bear markets via downward breakouts.
# Target: 20-50 total trades over 4 years (5-12/year) to minimize fee drag.

name = "1d_Donchian_Breakout_Volume_TrendFilter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 50-day EMA for trend filter
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 1d data for Donchian channels (already daily, but use for consistency)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period high/low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Rolling max/min for Donchian channels
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 1d timeframe (no shift needed as already aligned)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Volume confirmation: volume > 1.5 * 20-day average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure EMA and Donchian are valid
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_50[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above Donchian high with volume confirmation and uptrend
            if (close[i] > donchian_high_aligned[i] and volume_confirm[i] and close[i] > ema_50[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with volume confirmation and downtrend
            elif (close[i] < donchian_low_aligned[i] and volume_confirm[i] and close[i] < ema_50[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below Donchian low (trend reversal)
            if close[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above Donchian high (trend reversal)
            if close[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals