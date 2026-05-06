#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6s strategy using 12h Donchian channels and 1d EMA to capture breakouts with trend alignment
# - Uses 12h Donchian channels (20-period) to identify breakout levels
# - Uses 1d EMA (50) to determine long-term trend direction
# - Enters long when price breaks above 12h Donchian upper channel AND price > 1d EMA50
# - Enters short when price breaks below 12h Donchian lower channel AND price < 1d EMA50
# - Requires volume confirmation (volume > 1.5x 20-period average)
# - Exits when price returns to the 12h Donchian middle (mean of upper/lower)
# - Designed to work in both bull and bear markets by aligning with 1d trend
# - Target: 60-120 total trades over 4 years (15-30/year) with 0.25 position sizing

name = "6h_Donchian_EMA50_Trend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian channel calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Get 1d data for EMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 12h Donchian channels (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Upper channel (20-period high)
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    # Lower channel (20-period low)
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    # Middle channel (average of upper and lower)
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate 1d EMA (50)
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h indicators to 6h timeframe
    donchian_high_6h = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_6h = align_htf_to_ltf(prices, df_12h, donchian_low)
    donchian_mid_6h = align_htf_to_ltf(prices, df_12h, donchian_mid)
    
    # Align 1d EMA to 6h timeframe
    ema_50_6h = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume filter (6h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)  # Volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup
        # Skip if any critical value is NaN
        if (np.isnan(donchian_high_6h[i]) or np.isnan(donchian_low_6h[i]) or 
            np.isnan(donchian_mid_6h[i]) or np.isnan(ema_50_6h[i]) or
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above Donchian high + price above EMA50 + volume confirmation
            if close[i] > donchian_high_6h[i] and close[i] > ema_50_6h[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low + price below EMA50 + volume confirmation
            elif close[i] < donchian_low_6h[i] and close[i] < ema_50_6h[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian middle
            if close[i] <= donchian_mid_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian middle
            if close[i] >= donchian_mid_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals