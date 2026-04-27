#!/usr/bin/env python3
"""
4h Camarilla Pivot Breakout with 1d Trend and Volume Spike.
Long when price breaks above Camarilla R3 + 1d trend up + volume spike.
Short when price breaks below Camarilla S3 + 1d trend down + volume spike.
Exit when price returns to Camarilla Pivot or trend reverses.
Designed for low frequency (20-40 trades/year) to minimize fee drain.
Uses Camarilla pivots for structure, 1d EMA for trend, volume spike for confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_1d = np.empty_like(close_1d, dtype=np.float64)
    ema_1d.fill(np.nan)
    for i in range(33, len(close_1d)):
        ema_1d[i] = np.mean(close_1d[i-33:i+1])  # Simple MA for EMA approximation
    
    # Align 1d EMA to 4h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Camarilla pivot levels from previous 1d candle
    # Pivot = (H + L + C) / 3
    # R3 = Close + (H - L) * 1.1/2
    # S3 = Close - (H - L) * 1.1/2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    pivot = np.empty_like(close_1d_vals, dtype=np.float64)
    r3 = np.empty_like(close_1d_vals, dtype=np.float64)
    s3 = np.empty_like(close_1d_vals, dtype=np.float64)
    
    for i in range(len(close_1d_vals)):
        pivot[i] = (high_1d[i] + low_1d[i] + close_1d_vals[i]) / 3.0
        r3[i] = close_1d_vals[i] + (high_1d[i] - low_1d[i]) * 1.1 / 2.0
        s3[i] = close_1d_vals[i] - (high_1d[i] - low_1d[i]) * 1.1 / 2.0
    
    # Align pivot levels to 4h timeframe (use previous day's levels)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume filter: volume > 1.5x average (to avoid false breakouts)
    vol_ma_20 = np.empty_like(volume, dtype=np.float64)
    vol_ma_20.fill(np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need volume MA (20)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        
        # Current indicators
        pivot_val = pivot_aligned[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        trend_1d = ema_1d_aligned[i]
        
        # Volume filter: volume > 1.5x average
        vol_filter = vol_now > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Bull: price breaks above R3 + 1d trend up + volume spike
            if price_now > r3_val and price_now > trend_1d and vol_filter:
                signals[i] = size
                position = 1
            # Bear: price breaks below S3 + 1d trend down + volume spike
            elif price_now < s3_val and price_now < trend_1d and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to pivot or 1d trend turns down
            if price_now < pivot_val or price_now < trend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to pivot or 1d trend turns up
            if price_now > pivot_val or price_now > trend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0