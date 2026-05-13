#!/usr/bin/env python3
# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and 1d volume spike confirmation.
# Long when price breaks above Camarilla R3 AND 4h EMA50 uptrend AND 1d volume > 2.0 * 20-period average volume.
# Short when price breaks below Camarilla S3 AND 4h EMA50 downtrend AND 1d volume > 2.0 * 20-period average volume.
# Exit when price reverts to Camarilla pivot (PP) or opposite stop loss via signal=0.
# Uses discrete position sizing (0.20) to limit fee churn. Designed for 1h timeframe with HTF filters to reduce overtrading.
# Target: 60-150 total trades over 4 years = 15-37/year for 1h.

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_1dVolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 4h EMA50 for trend filter (HTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1d volume spike filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (2.0 * vol_ma_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    # Calculate Camarilla pivot levels (PP, R3, S3) from prior day
    # Prior day's high, low, close
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: PP = (H+L+C)/3, R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    camarilla_pp = (high_1d + low_1d + close_1d) / 3.0
    camarilla_r3 = close_1d + (high_1d - low_1d) * 1.1 / 2.0
    camarilla_s3 = close_1d - (high_1d - low_1d) * 1.1 / 2.0
    
    # Align Camarilla levels to 1h timeframe (use prior day's levels for current day)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):  # Start after first bar to have prior Camarilla levels
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or
            np.isnan(camarilla_pp_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = prices.index[i].hour
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above Camarilla R3 AND 4h EMA50 uptrend AND volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_50_4h_aligned[i] and  # price above EMA50 = uptrend
                volume_spike_aligned[i] > 0.5):  # True if volume spike aligned
                signals[i] = 0.20
                position = 1
            # SHORT: price breaks below Camarilla S3 AND 4h EMA50 downtrend AND volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_50_4h_aligned[i] and  # price below EMA50 = downtrend
                  volume_spike_aligned[i] > 0.5):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price reverts to Camarilla PP or breaks below S3 (stop)
            if close[i] <= camarilla_pp_aligned[i] or close[i] < camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: price reverts to Camarilla PP or breaks above R3 (stop)
            if close[i] >= camarilla_pp_aligned[i] or close[i] > camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals