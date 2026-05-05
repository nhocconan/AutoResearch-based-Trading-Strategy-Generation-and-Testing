#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Long when: price breaks above R3 (Camarilla resistance) AND 1d EMA34 uptrend (price > EMA34) AND volume > 2x 20-period MA
# Short when: price breaks below S3 (Camarilla support) AND 1d EMA34 downtrend (price < EMA34) AND volume > 2x 20-period MA
# Exit when: price reverts to Camarilla pivot point (PP) OR volume drops below average
# Uses Camarilla for intraday structure, 1d EMA for regime filter, volume for conviction
# Timeframe: 6h, HTF: 1d for EMA34. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "6h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels using previous day's OHLC (1d timeframe)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla pivot point
    pp = (prev_high + prev_low + prev_close) / 3
    # Camarilla R3 and S3 levels
    r3 = pp + (prev_high - prev_low) * 1.1 / 4
    s3 = pp - (prev_high - prev_low) * 1.1 / 4
    
    # Align 1d Camarilla levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate 1d EMA34 for trend filter
    if len(df_1d) >= 34:
        ema_34 = pd.Series(df_1d['close']).ewm(span=34, min_periods=34, adjust=False).mean().values
    else:
        ema_34 = np.full(len(df_1d), np.nan)
    
    # Align 1d EMA34 to 6h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume confirmation on 6h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R3 AND 1d EMA34 uptrend AND volume spike
            if (close[i] > r3_aligned[i] and 
                close[i] > ema_34_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below S3 AND 1d EMA34 downtrend AND volume spike
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema_34_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price reverts to pivot point OR volume drops below average
            if (close[i] <= pp_aligned[i] or not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reverts to pivot point OR volume drops below average
            if (close[i] >= pp_aligned[i] or not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals