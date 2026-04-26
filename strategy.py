#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_v1
Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and 6h volume spike confirmation.
In ranging markets, price reverts to mean at R3/S3 levels. In trending markets, breaks of R4/S4
indicate strong momentum. Volume spike confirms conviction. Works in both bull/bear by using 1d trend.
Target: 50-150 total trades over 4 years (12-37/year) via tight entry requiring Camarilla level,
trend alignment, and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF trend and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate daily Camarilla levels from previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla pivot = (high + low + close) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    # Camarilla R3, S3, R4, S4
    range_1d = high_1d - low_1d
    camarilla_r3 = close_1d + 1.1 * range_1d / 6
    camarilla_s3 = close_1d - 1.1 * range_1d / 6
    camarilla_r4 = close_1d + 1.1 * range_1d / 2
    camarilla_s4 = close_1d - 1.1 * range_1d / 2
    
    # Align Camarilla levels to 6h timeframe (completed daily bars only)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # 6h volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for EMA and 20 for volume MA)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(camarilla_r4_aligned[i]) or
            np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume spike condition (strict to reduce trades)
        volume_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        # Determine 1d trend: price above/below EMA34
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Long conditions: break above R3 in uptrend OR break above R4 (strong momentum)
        long_signal = False
        if uptrend and volume_spike:
            if close[i] > camarilla_r3_aligned[i]:
                long_signal = True
            elif close[i] > camarilla_r4_aligned[i]:  # Breakout of R4 = strong momentum
                long_signal = True
        
        # Short conditions: break below S3 in downtrend OR break below S4 (strong momentum)
        short_signal = False
        if downtrend and volume_spike:
            if close[i] < camarilla_s3_aligned[i]:
                short_signal = True
            elif close[i] < camarilla_s4_aligned[i]:  # Breakdown of S4 = strong momentum
                short_signal = True
        
        # Exit conditions: opposite Camarilla level or loss of volume/momentum
        exit_long = False
        exit_short = False
        if position == 1:
            # Exit long if price breaks below S3 or loses volume/uptrend
            if close[i] < camarilla_s3_aligned[i] or not (uptrend and volume_spike):
                exit_long = True
        elif position == -1:
            # Exit short if price breaks above R3 or loses volume/downtrend
            if close[i] > camarilla_r3_aligned[i] or not (downtrend and volume_spike):
                exit_short = True
        
        # Generate signals
        if exit_long:
            signals[i] = 0.0
            position = 0
        elif exit_short:
            signals[i] = 0.0
            position = 0
        elif long_signal and position != 1:
            signals[i] = 0.25
            position = 1
        elif long_signal and position == 1:
            signals[i] = 0.25
        elif short_signal and position != -1:
            signals[i] = -0.25
            position = -1
        elif short_signal and position == -1:
            signals[i] = -0.25
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0