#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h volume spike and 12h EMA50 trend filter
# Long when price breaks above Donchian upper AND volume > 2.0x 20-period average AND 12h EMA50 > EMA50_prev (uptrend)
# Short when price breaks below Donchian lower AND volume > 2.0x 20-period average AND 12h EMA50 < EMA50_prev (downtrend)
# Exit when price crosses back to Donchian midpoint OR 12h EMA50 flips direction
# Uses discrete sizing (0.25) to limit fee drag. Target: 20-50 trades/year per symbol.
# Donchian channels provide price structure, volume spike confirms institutional interest,
# 12h EMA50 filters for primary trend direction to avoid counter-trend whipsaws.
# Works in bull markets via longs in uptrends and bear markets via shorts in downtrends.

name = "4h_Donchian20_VolumeSpike_12hEMA50_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Donchian channels on 4h data (using lookback of 20 periods)
    if len(high) >= 20:
        # Rolling max/min for Donchian upper/lower
        roll_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
        roll_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
        # Midpoint for exit
        roll_mid = (roll_max + roll_min) / 2
    else:
        roll_max = np.full(n, np.nan)
        roll_min = np.full(n, np.nan)
        roll_mid = np.full(n, np.nan)
    
    # Get 12h data for EMA50 trend filter
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_prev = np.concatenate([[np.nan], ema_50[:-1]])  # Previous EMA for trend direction
    
    # Uptrend when current EMA50 > previous EMA50
    uptrend_12h = ema_50 > ema_50_prev
    downtrend_12h = ema_50 < ema_50_prev
    
    # Align 12h trend to 4h timeframe
    uptrend_12h_aligned = align_htf_to_ltf(prices, df_12h, uptrend_12h.astype(float))
    downtrend_12h_aligned = align_htf_to_ltf(prices, df_12h, downtrend_12h.astype(float))
    
    # Volume confirmation: volume > 2.0x 20-period average (spike filter)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(roll_max[i]) or 
            np.isnan(roll_min[i]) or 
            np.isnan(roll_mid[i]) or 
            np.isnan(uptrend_12h_aligned[i]) or 
            np.isnan(downtrend_12h_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper AND volume spike AND 12h uptrend
            if (close[i] > roll_max[i] and 
                volume_filter[i] and 
                uptrend_12h_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower AND volume spike AND 12h downtrend
            elif (close[i] < roll_min[i] and 
                  volume_filter[i] and 
                  downtrend_12h_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back to midpoint OR 12h trend flips to downtrend
            if (close[i] < roll_mid[i] or 
                downtrend_12h_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back to midpoint OR 12h trend flips to uptrend
            if (close[i] > roll_mid[i] or 
                uptrend_12h_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals