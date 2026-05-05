#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation (1.5x)
# Long when price breaks above Donchian upper band AND price > 1w EMA50 AND volume > 1.5x 20-period average
# Short when price breaks below Donchian lower band AND price < 1w EMA50 AND volume > 1.5x 20-period average
# Exit when price reverts to Donchian midpoint OR 1w EMA50 filter reverses
# Uses 1d timeframe for lower frequency trading (target: 30-100 trades over 4 years) to minimize fee drag
# 1w EMA50 provides higher timeframe trend filter effective in both bull and bear markets
# Donchian channels provide clear breakout levels with built-in volatility adjustment
# Designed for BTC/ETH with SOL as secondary confirmation

name = "1d_Donchian20_1wEMA50_VolumeSpike_1.5x"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate Donchian(20) on 1d (using previous 20 periods to avoid look-ahead)
    if len(high) >= 20:
        # Donchian upper band = highest high of previous 20 periods
        donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
        # Donchian lower band = lowest low of previous 20 periods
        donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
        # Donchian midpoint = (upper + lower) / 2
        donchian_mid = (donchian_high + donchian_low) / 2.0
    else:
        donchian_high = np.full(n, np.nan)
        donchian_low = np.full(n, np.nan)
        donchian_mid = np.full(n, np.nan)
    
    # Calculate 1w EMA(50)
    if len(close_1w) >= 50:
        ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    else:
        ema_50_1w = np.full(len(close_1w), np.nan)
    
    # Align HTF indicators to 1d timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation on 1d (threshold: 1.5x for optimal frequency)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (1.5 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper band AND price > EMA50 AND volume spike
            if (close[i] > donchian_high[i] and 
                close[i] > ema_50_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower band AND price < EMA50 AND volume spike
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema_50_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price reverts to Donchian midpoint OR price < EMA50 (trend weakening)
            if close[i] < donchian_mid[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reverts to Donchian midpoint OR price > EMA50 (trend weakening)
            if close[i] > donchian_mid[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals