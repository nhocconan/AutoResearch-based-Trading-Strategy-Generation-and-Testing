#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d ATR expansion + 1w Donchian breakout for trend-following in volatile markets
# Long when 1d ATR ratio (ATR7/ATR30) > 1.8 AND price breaks above 1w Donchian upper channel (20) AND volume > 1.5 * avg_volume(20)
# Short when 1d ATR ratio > 1.8 AND price breaks below 1w Donchian lower channel (20) AND volume > 1.5 * avg_volume(20)
# ATR expansion captures volatility spikes that often precede strong moves; Donchian provides directional structure
# Volume confirmation ensures breakout validity while limiting false signals
# Works in both bull (buy breakouts) and bear (sell breakdowns) markets
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe

name = "12h_1dATR_Expansion_1wDonchian_Breakout_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need at least 30 completed 1d bars for ATR30
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range for 1d
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First bar: no previous close
    tr2[0] = high_1d[0] - close_1d[0]  # Approximation for first bar
    tr3[0] = low_1d[0] - close_1d[0]   # Approximation for first bar
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate ATR7 and ATR30 for 1d
    atr7_1d = pd.Series(tr).rolling(window=7, min_periods=7).mean().values
    atr30_1d = pd.Series(tr).rolling(window=30, min_periods=30).mean().values
    atr_ratio_1d = np.where(atr30_1d > 0, atr7_1d / atr30_1d, 0)
    
    # Align 1d ATR ratio to 12h timeframe (wait for completed 1d bar)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    # Get 1w data ONCE before loop for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:  # Need at least 20 completed weekly bars for Donchian20
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Donchian channels (20-period)
    donchian_high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align 1w Donchian channels to 12h timeframe (wait for completed 1w bar)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_20)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 12h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(atr_ratio_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: ATR expansion + price breaks above Donchian high + volume spike, in session
            if (atr_ratio_aligned[i] > 1.8 and 
                close[i] > donchian_high_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: ATR expansion + price breaks below Donchian low + volume spike, in session
            elif (atr_ratio_aligned[i] > 1.8 and 
                  close[i] < donchian_low_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian low (trend reversal)
            if close[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Donchian high (trend reversal)
            if close[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals