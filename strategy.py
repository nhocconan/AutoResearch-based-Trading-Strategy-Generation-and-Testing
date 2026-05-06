#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Donchian channel breakout with volume confirmation and 1d trend filter
# - Uses 12h Donchian(20) breakout for entry signals
# - Confirms with volume spike (>2x 20-period average) on 6h timeframe
# - Uses 1d EMA(50) to filter trades in direction of higher timeframe trend
# - Exits when price returns to the 12h Donchian midpoint or reverses
# - Designed to capture medium-term breakouts with trend alignment to reduce false signals
# - Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "6h_12hDonchian_20_1dEMA50_VolumeBreakout"
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
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 12h Donchian channel (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Upper band: highest high of last 20 periods
    high_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low of last 20 periods
    low_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    # Middle band: average of upper and lower
    mid_20 = (high_20 + low_20) / 2
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h indicators to 6h timeframe
    high_20_6h = align_htf_to_ltf(prices, df_12h, high_20)
    low_20_6h = align_htf_to_ltf(prices, df_12h, low_20)
    mid_20_6h = align_htf_to_ltf(prices, df_12h, mid_20)
    ema_50_6h = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume filters (6h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)  # Volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup
        # Skip if any critical value is NaN
        if (np.isnan(high_20_6h[i]) or np.isnan(low_20_6h[i]) or np.isnan(mid_20_6h[i]) or
            np.isnan(ema_50_6h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 12h Donchian upper band with volume spike and above 1d EMA50
            if close[i] > high_20_6h[i] and volume_spike[i] and close[i] > ema_50_6h[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12h Donchian lower band with volume spike and below 1d EMA50
            elif close[i] < low_20_6h[i] and volume_spike[i] and close[i] < ema_50_6h[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to 12h Donchian midpoint or breaks below lower band
            if close[i] < mid_20_6h[i] or close[i] < low_20_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to 12h Donchian midpoint or breaks above upper band
            if close[i] > mid_20_6h[i] or close[i] > high_20_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals