#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume spike confirmation
# Long when price breaks above Donchian(20) high AND volume > 2.0x 20-period average AND 1w EMA50 uptrend
# Short when price breaks below Donchian(20) low AND volume > 2.0x 20-period average AND 1w EMA50 downtrend
# Exit when price crosses Donchian(20) midpoint OR 1w trend reverses
# Uses discrete sizing (0.25) to limit fee drag. Target: 7-25 trades/year per symbol.
# Donchian provides objective breakout levels, volume spike confirms institutional participation,
# 1w EMA50 filters for higher timeframe direction to avoid counter-trend whipsaws.
# Works in bull markets via longs in uptrends and bear markets via shorts in downtrends.

name = "1d_Donchian20_VolumeSpike_1wEMA50_Trend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Donchian Channel on 1d data (20-period high/low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Align Donchian levels to prices timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid)
    
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
    
    # Align 1w trend to 1d timeframe
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w.astype(float))
    downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, downtrend_1w.astype(float))
    
    # Volume confirmation: volume > 2.0x 20-period average (spike filter)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)  # No volume confirmation if insufficient data
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or 
            np.isnan(uptrend_1w_aligned[i]) or 
            np.isnan(downtrend_1w_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high AND volume spike AND 1w EMA50 uptrend
            if (close[i] > donchian_high_aligned[i] and 
                volume_filter[i] and 
                uptrend_1w_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low AND volume spike AND 1w EMA50 downtrend
            elif (close[i] < donchian_low_aligned[i] and 
                  volume_filter[i] and 
                  downtrend_1w_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian midpoint OR 1w trend changes to downtrend
            if (close[i] < donchian_mid_aligned[i] or 
                downtrend_1w_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Donchian midpoint OR 1w trend changes to uptrend
            if (close[i] > donchian_mid_aligned[i] or 
                uptrend_1w_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals