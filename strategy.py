#!/usr/bin/env python3
"""
12h Williams Alligator with 1d Trend Filter and Volume Spike.
Long when price above Alligator teeth (green line) + 1d trend up + volume spike.
Short when price below Alligator teeth + 1d trend down + volume spike.
Exit when price crosses Alligator jaw (blue line) or trend reverses.
Designed for low frequency (12-37 trades/year) to minimize fee drag on 12h timeframe.
Uses Williams Alligator (SMAs with specific periods) and 1d EMA for trend filter.
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_1d = np.empty_like(close_1d, dtype=np.float64)
    ema_1d.fill(np.nan)
    # Calculate EMA properly with alpha
    alpha = 2.0 / (50 + 1)
    for i in range(len(close_1d)):
        if i == 0:
            ema_1d[i] = close_1d[i]
        elif np.isnan(ema_1d[i-1]):
            ema_1d[i] = close_1d[i]
        else:
            ema_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_1d[i-1]
    
    # Align 1d EMA to 12h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Williams Alligator: Jaw (Blue) = SMA(13, 8), Teeth (Green) = SMA(8, 5), Lips (Red) = SMA(5, 3)
    # Calculate SMAs with proper handling
    jaw_period = 13
    jaw_shift = 8
    teeth_period = 8
    teeth_shift = 5
    lips_period = 5
    lips_shift = 3
    
    # Jaw (Blue line)
    jaw = np.full(n, np.nan)
    for i in range(jaw_period - 1 + jaw_shift, n):
        start_idx = i - jaw_shift - jaw_period + 1
        end_idx = i - jaw_shift + 1
        if start_idx >= 0:
            jaw[i] = np.mean(close[start_idx:end_idx])
    
    # Teeth (Green line)
    teeth = np.full(n, np.nan)
    for i in range(teeth_period - 1 + teeth_shift, n):
        start_idx = i - teeth_shift - teeth_period + 1
        end_idx = i - teeth_shift + 1
        if start_idx >= 0:
            teeth[i] = np.mean(close[start_idx:end_idx])
    
    # Lips (Red line)
    lips = np.full(n, np.nan)
    for i in range(lips_period - 1 + lips_shift, n):
        start_idx = i - lips_shift - lips_period + 1
        end_idx = i - lips_shift + 1
        if start_idx >= 0:
            lips[i] = np.mean(close[start_idx:end_idx])
    
    # Volume filter: volume > 2.0x average (to avoid false breakouts)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Alligator lines + 1d EMA + volume MA (20)
    start_idx = max(
        jaw_period - 1 + jaw_shift,
        teeth_period - 1 + teeth_shift,
        lips_period - 1 + lips_shift,
        19,  # volume MA
        50   # 1d EMA
    )
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        
        # Current indicators
        jaw_val = jaw[i]
        teeth_val = teeth[i]  # Green line - main signal
        lips_val = lips[i]
        trend_1d = ema_1d_aligned[i]
        
        # Volume filter: volume > 2.0x average
        vol_filter = vol_now > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Bull: price above teeth (green) + 1d trend up + volume spike
            if price_now > teeth_val and price_now > trend_1d and vol_filter:
                signals[i] = size
                position = 1
            # Bear: price below teeth (green) + 1d trend down + volume spike
            elif price_now < teeth_val and price_now < trend_1d and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below jaw (blue line) or 1d trend turns down
            if price_now < jaw_val or price_now < trend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above jaw (blue line) or 1d trend turns up
            if price_now > jaw_val or price_now > trend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_WilliamsAlligator_1dEMA50_Volume"
timeframe = "12h"
leverage = 1.0