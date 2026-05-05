#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Long when price breaks above Donchian upper band AND volume > 1.5x 20-period average AND 1w EMA50 uptrend
# Short when price breaks below Donchian lower band AND volume > 1.5x 20-period average AND 1w EMA50 downtrend
# Exit when price crosses Donchian middle band (20-period mean) OR 1w trend reverses
# Uses discrete sizing (0.25) to limit fee drag. Target: 12-30 trades/year per symbol.
# Donchian channels provide clear breakout levels, volume spike confirms institutional participation,
# 1w EMA50 filters for higher timeframe direction to avoid counter-trend whipsaws in bear markets.
# Works in bull markets via longs in uptrends and bear markets via shorts in downtrends.

name = "12h_Donchian20_VolumeSpike_1wEMA50_Trend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for Donchian calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels on 12h data (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Donchian upper band: 20-period high
    donchian_upper = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    # Donchian lower band: 20-period low
    donchian_lower = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    # Donchian middle band: 20-period mean
    donchian_middle = (pd.Series(high_12h).rolling(window=20, min_periods=20).mean().values + 
                       pd.Series(low_12h).rolling(window=20, min_periods=20).mean().values) / 2
    
    # Align Donchian levels to prices timeframe
    upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper)
    lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower)
    middle_aligned = align_htf_to_ltf(prices, df_12h, donchian_middle)
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Uptrend when close > EMA50, downtrend when close < EMA50
    uptrend_1w = close_1w > ema_50_1w
    downtrend_1w = close_1w < ema_50_1w
    
    # Align 1w trend to 12h timeframe
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w.astype(float))
    downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, downtrend_1w.astype(float))
    
    # Volume confirmation: volume > 1.5x 20-period average (spike filter)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)  # No volume confirmation if insufficient data
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(upper_aligned[i]) or 
            np.isnan(lower_aligned[i]) or 
            np.isnan(middle_aligned[i]) or 
            np.isnan(uptrend_1w_aligned[i]) or 
            np.isnan(downtrend_1w_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper band AND volume spike AND 1w EMA50 uptrend
            if (close[i] > upper_aligned[i] and 
                volume_filter[i] and 
                uptrend_1w_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower band AND volume spike AND 1w EMA50 downtrend
            elif (close[i] < lower_aligned[i] and 
                  volume_filter[i] and 
                  downtrend_1w_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian middle band OR 1w trend changes to downtrend
            if (close[i] < middle_aligned[i] or 
                downtrend_1w_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Donchian middle band OR 1w trend changes to uptrend
            if (close[i] > middle_aligned[i] or 
                uptrend_1w_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals