#!/usr/bin/env python3
"""
4h Camarilla Pivot R3/S3 Breakout with Volume Spike and 1h Trend Filter.
Long when price breaks above R3 with volume > 2x average and 1h EMA50 up.
Short when price breaks below S3 with volume > 2x average and 1h EMA50 down.
Exit when price returns to pivot point (PP) or reverses with volume.
Designed for 20-40 trades/year per symbol with strong institutional levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period"""
    range_val = high - low
    pp = (high + low + close) / 3.0
    r1 = pp + (range_val * 1.1 / 12)
    s1 = pp - (range_val * 1.1 / 12)
    r2 = pp + (range_val * 1.1 / 6)
    s2 = pp - (range_val * 1.1 / 6)
    r3 = pp + (range_val * 1.1 / 4)
    s3 = pp - (range_val * 1.1 / 4)
    r4 = pp + (range_val * 1.1 / 2)
    s4 = pp - (range_val * 1.1 / 2)
    return pp, r1, r2, r3, r4, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pp, r1, r2, r3, r4, s1, s2, s3, s4 = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Align to 4h timeframe (wait for daily close)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Get 1h data for trend filter (EMA50)
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 1:
        return np.zeros(n)
    
    # Calculate 1h EMA50
    close_1h = df_1h['close'].values
    ema_50 = pd.Series(close_1h).ewm(span=50, adjust=False, min_periods=50).values
    ema_50_aligned = align_htf_to_ltf(prices, df_1h, ema_50)
    
    # Volume filter: volume > 2x average
    vol_ma_20 = np.empty_like(volume, dtype=np.float64)
    vol_ma_20.fill(np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Camarilla (1d) + EMA50 (1h) + volume MA (20)
    start_idx = max(19, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        
        # Current levels
        pp_val = pp_aligned[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        ema_50_val = ema_50_aligned[i]
        
        # Volume filter: volume > 2x average
        vol_filter = vol_now > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above R3 with volume and 1h uptrend
            if price_now > r3_val and vol_filter and ema_50_val > ema_50_val:  # EMA50 rising (simplified)
                # Actually check if EMA is rising compared to previous
                if i > start_idx and ema_50_aligned[i] > ema_50_aligned[i-1]:
                    signals[i] = size
                    position = 1
            # Short: price breaks below S3 with volume and 1h downtrend
            elif price_now < s3_val and vol_filter and ema_50_val < ema_50_val:  # EMA50 falling
                if i > start_idx and ema_50_aligned[i] < ema_50_aligned[i-1]:
                    signals[i] = -size
                    position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to PP or reverses with volume
            if price_now <= pp_val or (price_now < ema_50_val and vol_filter):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to PP or reverses with volume
            if price_now >= pp_val or (price_now > ema_50_val and vol_filter):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_Volume_1hEMA50"
timeframe = "4h"
leverage = 1.0