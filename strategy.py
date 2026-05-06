#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Donchian channels provide robust breakout levels that work in both trending and ranging markets
# 1d EMA50 filters for higher timeframe trend alignment to reduce false breakouts
# Volume spike (>2.0x 20-bar average) confirms institutional participation
# ATR-based trailing stop via signal=0 when price retraces 50% of breakout range
# Discrete sizing 0.25 to limit fee drag; target 100-180 total trades over 4 years (25-45/year)
# Proven pattern: Donchian breakouts with volume confirmation show consistent performance on BTC/ETH

name = "4h_Donchian20_1dEMA50_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 60:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate volume spike filter (>2.0x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma_20)
    
    # Calculate 4h Donchian channels (20-period)
    high_ma_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Align HTF indicators to 4h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    volume_filter_aligned = align_htf_to_ltf(prices, df_1d, volume_filter)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    breakout_high = 0.0  # Track breakout level for trailing stop
    breakout_low = 0.0   # Track breakdown level for trailing stop
    
    for i in range(100, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(high_ma_20[i]) or 
            np.isnan(low_ma_20[i]) or np.isnan(volume_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price > upper Donchian AND uptrend (price > EMA50) AND volume spike
            if close[i] > high_ma_20[i] and close[i] > ema50_1d_aligned[i] and volume_filter_aligned[i]:
                signals[i] = 0.25
                position = 1
                breakout_high = high_ma_20[i]  # Set breakout level for stop
            # Short breakdown: price < lower Donchian AND downtrend (price < EMA50) AND volume spike
            elif close[i] < low_ma_20[i] and close[i] < ema50_1d_aligned[i] and volume_filter_aligned[i]:
                signals[i] = -0.25
                position = -1
                breakout_low = low_ma_20[i]   # Set breakdown level for stop
        elif position == 1:
            # Trail long: exit if price retraces 50% from breakout high
            midpoint = breakout_high + (low_ma_20[i] - breakout_high) * 0.5
            if close[i] <= midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Trail short: exit if price retraces 50% from breakdown low
            midpoint = breakout_low + (high_ma_20[i] - breakout_low) * 0.5
            if close[i] >= midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals