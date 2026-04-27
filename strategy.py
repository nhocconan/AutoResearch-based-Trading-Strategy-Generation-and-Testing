#!/usr/bin/env python3
"""
12h Williams Alligator + 1d EMA50 Trend + Volume Spike.
Williams Alligator uses three SMAs (Jaw:13, Teeth:8, Lips:5) with future shift.
Long when Lips > Teeth > Jaw (bullish alignment) + price above EMA50 + volume spike.
Short when Jaw > Teeth > Lips (bearish alignment) + price below EMA50 + volume spike.
Exit when Alligator alignment breaks or price crosses EMA50.
Designed for low-frequency, high-conviction trades in trending markets.
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
    
    # Get daily data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate daily EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_1d = np.empty_like(close_1d, dtype=np.float64)
    ema_1d.fill(np.nan)
    alpha = 2.0 / (50 + 1)
    for i in range(len(close_1d)):
        if i == 0:
            ema_1d[i] = close_1d[i]
        elif np.isnan(ema_1d[i-1]):
            ema_1d[i] = close_1d[i]
        else:
            ema_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_1d[i-1]
    
    # Align daily EMA to 12h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Get 12h data for Williams Alligator
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 12h: SMAs with future shift
    close_12h = df_12h['close'].values
    
    # Jaw: 13-period SMA, shifted by 8 bars
    jaw = np.full_like(close_12h, np.nan)
    for i in range(12, len(close_12h)):
        jaw[i] = np.mean(close_12h[i-12:i+1])
    jaw = np.roll(jaw, 8)  # shift future data to the right
    
    # Teeth: 8-period SMA, shifted by 5 bars
    teeth = np.full_like(close_12h, np.nan)
    for i in range(7, len(close_12h)):
        teeth[i] = np.mean(close_12h[i-7:i+1])
    teeth = np.roll(teeth, 5)
    
    # Lips: 5-period SMA, shifted by 3 bars
    lips = np.full_like(close_12h, np.nan)
    for i in range(4, len(close_12h)):
        lips[i] = np.mean(close_12h[i-4:i+1])
    lips = np.roll(lips, 3)
    
    # Align Alligator lines to 12h timeframe (no additional delay needed as SMAs are on close)
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Volume filter: volume > 2.0x average (to avoid false signals)
    vol_ma_20 = np.empty_like(volume, dtype=np.float64)
    vol_ma_20.fill(np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need daily EMA (50) + Alligator (max 13+8=21) + volume MA (20)
    start_idx = max(20, 21)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        
        # Current indicators
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        trend_1d = ema_1d_aligned[i]
        
        # Volume filter: volume > 2.0x average
        vol_filter = vol_now > 2.0 * vol_ma_20[i]
        
        # Alligator alignment
        bullish_alignment = lips_val > teeth_val and teeth_val > jaw_val
        bearish_alignment = jaw_val > teeth_val and teeth_val > lips_val
        
        if position == 0:
            # Bull: Lips > Teeth > Jaw + price above EMA50 + volume spike
            if bullish_alignment and price_now > trend_1d and vol_filter:
                signals[i] = size
                position = 1
            # Bear: Jaw > Teeth > Lips + price below EMA50 + volume spike
            elif bearish_alignment and price_now < trend_1d and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Alligator alignment breaks or price crosses below EMA50
            if not bullish_alignment or price_now < trend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Alligator alignment breaks or price crosses above EMA50
            if not bearish_alignment or price_now > trend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_WilliamsAlligator_1dEMA50_Volume"
timeframe = "12h"
leverage = 1.0