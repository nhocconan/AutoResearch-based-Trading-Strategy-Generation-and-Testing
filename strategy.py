#!/usr/bin/env python3
"""
4h Camarilla R3-S3 Breakout with 1d EMA34 Trend Filter and Volume Spike
Long when price breaks above R3 with 1d EMA34 uptrend and volume > 1.5x average.
Short when price breaks below S3 with 1d EMA34 downtrend and volume > 1.5x average.
Exit when price reverts to pivot point (P) level.
Designed to capture strong momentum moves while filtering choppy markets.
Target: 20-50 trades per year on 4h timeframe.
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 for trend filter
    ema_period = 34
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= ema_period:
        ema_1d[ema_period - 1] = np.mean(close_1d[:ema_period])
        for i in range(ema_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * (2 / (ema_period + 1)) + 
                         ema_1d[i - 1] * (1 - (2 / (ema_period + 1))))
    
    # Calculate Camarilla pivot levels for previous day (requires high, low, close)
    # Need at least 2 days of data to compute pivots for current day
    pivot_p = np.full(n, np.nan)
    r3 = np.full(n, np.nan)
    s3 = np.full(n, np.nan)
    
    for i in range(1, n):
        # Use previous day's OHLC to calculate today's Camarilla levels
        phigh = df_1d['high'].values[i-1] if i-1 < len(df_1d) else np.nan
        plow = df_1d['low'].values[i-1] if i-1 < len(df_1d) else np.nan
        pclose = df_1d['close'].values[i-1] if i-1 < len(df_1d) else np.nan
        
        if not (np.isnan(phigh) or np.isnan(plow) or np.isnan(pclose)):
            pivot_p[i] = (phigh + plow + pclose) / 3
            range_val = phigh - plow
            r3[i] = pclose + range_val * 1.1 * 6 / 8  # R3 = C + (H-L)*1.1*6/8
            s3[i] = pclose - range_val * 1.1 * 6 / 8  # S3 = C - (H-L)*1.1*6/8
    
    # Volume MA for confirmation (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i - 19:i + 1])
    
    # Align 1d EMA to 4h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need at least 1 day of pivot data, EMA34, and volume MA20
    start_idx = max(1, ema_period - 1, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(pivot_p[i]) or np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: require volume above average
        vol_filter = vol_now > 1.5 * vol_avg
        
        if position == 0:
            # Long: price breaks above R3 with 1d EMA34 uptrend and volume filter
            if (price > r3[i] and 
                close[i-1] <= r3[i] and  # Ensure we're breaking above (not already above)
                price > ema_1d_aligned[i] and vol_filter):
                signals[i] = size
                position = 1
            # Short: price breaks below S3 with 1d EMA34 downtrend and volume filter
            elif (price < s3[i] and 
                  close[i-1] >= s3[i] and  # Ensure we're breaking below (not already below)
                  price < ema_1d_aligned[i] and vol_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to pivot point (mean reversion)
            if price <= pivot_p[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to pivot point (mean reversion)
            if price >= pivot_p[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0