#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h trend filter and volume spike confirmation
# Designed for 25-35 trades/year with proper risk control via trend failure
# Long: price breaks above Donchian(20) high + price > 12h EMA50 + volume spike
# Short: price breaks below Donchian(20) low + price < 12h EMA50 + volume spike
# Exit: trend failure (price crosses 12h EMA50) or opposite breakout
# Volume filter: current 4h volume > 1.5x 20-period average to avoid false breakouts
# Donchian provides clear trend structure, EMA50 on 12h filters trend direction, volume confirms breakout strength

name = "4h_Donchian_Breakout_12hEMA50_VolumeFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Donchian channels (20-period) on 4h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period average volume for volume filter
    volume_series = pd.Series(volume)
    vol_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Align 12h EMA50 to 4h timeframe
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup period
    
    for i in range(start_idx, n):
        if np.isnan(ema50_12h_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current 4h volume > 1.5x 20-period average
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Look for breakout with trend and volume confirmation
            # Long: price breaks above Donchian high + uptrend + volume spike
            if close[i] > donchian_high[i] and ema50_12h_aligned[i] > 0:
                if vol_filter:
                    signals[i] = 0.25
                    position = 1
            # Short: price breaks below Donchian low + downtrend + volume spike
            elif close[i] < donchian_low[i] and ema50_12h_aligned[i] < 0:
                if vol_filter:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: trend failure (price crosses below EMA50) or opposite breakout
            if ema50_12h_aligned[i] <= 0 or close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: trend failure (price crosses above EMA50) or opposite breakout
            if ema50_12h_aligned[i] >= 0 or close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals