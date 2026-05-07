#!/usr/bin/env python3
# 1H_Camarilla_R3_S3_1DTrend_VolumeSpike
# Hypothesis: Combines 1-day Camarilla R3/S3 breakout with 4-hour EMA trend filter and volume spike confirmation on 1h timeframe.
# Uses daily pivot levels for institutional support/resistance, 4h EMA50 for trend alignment, and volume spikes to filter false breakouts.
# Designed for 1h timeframe with moderate trade frequency (15-37/year) and performance in both bull and bear regimes.
# Target: 60-150 total trades over 4 years with clear entry/exit rules.

name = "1H_Camarilla_R3_S3_1DTrend_VolumeSpike"
timeframe = "1h"
leverage = 1.0

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
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily OHLC for Camarilla pivots
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R3 = close + (high - low) * 1.1/4, S3 = close - (high - low) * 1.1/4
    rng = high_1d - low_1d
    camarilla_r3 = close_1d + rng * 1.1 / 4
    camarilla_s3 = close_1d - rng * 1.1 / 4
    
    # 4-hour EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all 1d and 4h data to 1h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    ema50_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Volume filter: current volume > 2.0x average volume (24-period)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(24, 50)  # Ensure we have volume MA and EMA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: spike confirmation
        volume_filter = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Price breaks above R3 + Uptrend (price > EMA50) + volume spike
            if (close[i] > r3_aligned[i] and 
                close[i] > ema50_aligned[i] and
                volume_filter):
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below S3 + Downtrend (price < EMA50) + volume spike
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema50_aligned[i] and
                  volume_filter):
                signals[i] = -0.20
                position = -1
        elif position != 0:
            # Exit: Price returns inside pivot range (reversion to mean)
            price_inside = (close[i] < r3_aligned[i] and close[i] > s3_aligned[i])
            
            if price_inside:
                signals[i] = 0.0
                position = 0
            else:
                # Maintain position
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals