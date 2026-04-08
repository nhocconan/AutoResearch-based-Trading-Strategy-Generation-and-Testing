#!/usr/bin/env python3
# 12h_1w_1d_camarilla_pivot_v1
# Hypothesis: 12-hour Camarilla pivot strategy with weekly trend filter and daily volume confirmation.
# Long when price touches S3 with bullish weekly trend and above-average volume.
# Short when price touches R3 with bearish weekly trend and above-average volume.
# Exit when price crosses the pivot point (PP).
# Uses weekly trend for direction, daily volume for confirmation, and Camarilla levels for precise entries.
# Target: 12-37 trades/year to stay within optimal range for 12h timeframe.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_1d_camarilla_pivot_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate daily volume average for confirmation
    vol_ma = np.zeros(n)
    vol_ma[:] = np.nan
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA20 for trend
    ema_1w_20 = np.zeros(len(close_1w))
    ema_1w_20[:] = np.nan
    if len(close_1w) >= 20:
        ema_1w_20[19] = np.mean(close_1w[:20])
        for i in range(20, len(close_1w)):
            ema_1w_20[i] = close_1w[i] * 0.0952 + ema_1w_20[i-1] * 0.9048
    ema_1w_20_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_20)
    
    # Get daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each day
    # PP = (H + L + C) / 3
    # S1 = C - (H - L) * 1.1 / 12
    # S2 = C - (H - L) * 1.1 / 6
    # S3 = C - (H - L) * 1.1 * 0.5
    # R3 = C + (H - L) * 1.1 * 0.5
    # R2 = C + (H - L) * 1.1 / 6
    # R1 = C + (H - L) * 1.1 / 12
    camarilla_pp = np.zeros(len(close_1d))
    camarilla_s3 = np.zeros(len(close_1d))
    camarilla_r3 = np.zeros(len(close_1d))
    
    for i in range(len(close_1d)):
        h = high_1d[i]
        l = low_1d[i]
        c = close_1d[i]
        pp = (h + l + c) / 3.0
        camarilla_pp[i] = pp
        camarilla_s3[i] = c - (h - l) * 1.1 * 0.5
        camarilla_r3[i] = c + (h - l) * 1.1 * 0.5
    
    # Align Camarilla levels to 12h timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup period
        price = close[i]
        vol_ma_val = vol_ma[i]
        weekly_trend = ema_1w_20_aligned[i]
        pp_level = camarilla_pp_aligned[i]
        s3_level = camarilla_s3_aligned[i]
        r3_level = camarilla_r3_aligned[i]
        
        # Skip if any values are not ready
        if (np.isnan(vol_ma_val) or np.isnan(weekly_trend) or 
            np.isnan(pp_level) or np.isnan(s3_level) or np.isnan(r3_level)):
            if position != 0:
                pass  # Hold current position
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 20-period average
        volume_ok = volume[i] > vol_ma_val
        
        if position == 1:  # Long position
            # Exit: price crosses above pivot point (PP)
            if price > pp_level:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price crosses below pivot point (PP)
            if price < pp_level:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat - look for new entries
            # Long entry: price at or below S3 with bullish weekly trend and volume confirmation
            if (price <= s3_level and 
                price > weekly_trend and  # Price above weekly EMA20 = bullish trend
                volume_ok):
                position = 1
                signals[i] = 0.25
            # Short entry: price at or above R3 with bearish weekly trend and volume confirmation
            elif (price >= r3_level and 
                  price < weekly_trend and  # Price below weekly EMA20 = bearish trend
                  volume_ok):
                position = -1
                signals[i] = -0.25
    
    return signals