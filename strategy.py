#!/usr/bin/env python3
"""
1h Bollinger Band Breakout with 4h Trend Filter and Volume Spike.
Long when price breaks above upper BB (20,2) + 4h trend up + volume spike.
Short when price breaks below lower BB (20,2) + 4h trend down + volume spike.
Exit when price returns to middle band (20) or trend reverses.
Designed for low frequency (15-35 trades/year) to minimize fee drag.
Uses Bollinger Bands for volatility breakout and 4h EMA for trend filter.
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
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h EMA(34) for trend filter
    close_4h = df_4h['close'].values
    ema_4h = np.empty_like(close_4h, dtype=np.float64)
    ema_4h.fill(np.nan)
    for i in range(33, len(close_4h)):
        ema_4h[i] = np.mean(close_4h[i-33:i+1])  # Simple MA for EMA approximation
    
    # Align 4h EMA to 1h timeframe
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate Bollinger Bands (20,2) on 1h
    bb_middle = np.empty_like(close, dtype=np.float64)
    bb_std = np.empty_like(close, dtype=np.float64)
    bb_middle.fill(np.nan)
    bb_std.fill(np.nan)
    for i in range(19, n):
        bb_middle[i] = np.mean(close[i-19:i+1])
        bb_std[i] = np.std(close[i-19:i+1])
    
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    # Volume filter: volume > 1.5x average (to avoid false breakouts)
    vol_ma_20 = np.empty_like(volume, dtype=np.float64)
    vol_ma_20.fill(np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.20   # 20% position size
    
    # Warmup: need BB (20) + volume MA (20) + 4h EMA (34)
    start_idx = max(20, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(bb_middle[i]) or np.isnan(ema_4h_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        
        # Current indicators
        bb_up = bb_upper[i]
        bb_low = bb_lower[i]
        bb_mid = bb_middle[i]
        trend_4h = ema_4h_aligned[i]
        
        # Volume filter: volume > 1.5x average
        vol_filter = vol_now > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Bull: price breaks above upper BB + 4h trend up + volume spike
            if price_now > bb_up and price_now > trend_4h and vol_filter:
                signals[i] = size
                position = 1
            # Bear: price breaks below lower BB + 4h trend down + volume spike
            elif price_now < bb_low and price_now < trend_4h and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to middle BB or 4h trend turns down
            if price_now < bb_mid or price_now < trend_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to middle BB or 4h trend turns up
            if price_now > bb_mid or price_now > trend_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_BollingerBreakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0