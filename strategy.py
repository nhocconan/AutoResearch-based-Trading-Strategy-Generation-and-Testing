#!/usr/bin/env python3
"""
4h Camarilla R1/S1 Breakout with 1d Trend Filter and Volume Spike.
Long when price breaks above R1 + 1d trend up + volume spike.
Short when price breaks below S1 + 1d trend down + volume spike.
Exit when price returns to pivot or trend reverses.
Designed for low frequency (20-50 trades/year) to minimize fee drag.
Uses Camarilla pivot levels and 1d EMA for trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for pivot levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
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
    # R1 = close + (high - low) * 1.12
    # S1 = close - (high - low) * 1.12
    # Pivot = (high + low + close) / 3
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    # Previous day's values (shifted by 1 to avoid look-ahead)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d_vals, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate pivot levels
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.12
    camarilla_pivot = (prev_high + prev_low + prev_close) / 3
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    # Volume filter: volume > 1.5x average (to avoid false breakouts)
    vol_ma_30 = np.empty_like(volume, dtype=np.float64)
    vol_ma_30.fill(np.nan)
    for i in range(29, n):
        vol_ma_30[i] = np.mean(volume[i-29:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need volume MA (30) + Camarilla levels (1d)
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(vol_ma_30[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        
        # Current indicators
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        pivot = camarilla_pivot_aligned[i]
        trend_1d = ema_1d_aligned[i]
        
        # Volume filter: volume > 1.5x average
        vol_filter = vol_now > 1.5 * vol_ma_30[i]
        
        if position == 0:
            # Bull: price breaks above R1 + 1d trend up + volume spike
            if price_now > r1 and price_now > trend_1d and vol_filter:
                signals[i] = size
                position = 1
            # Bear: price breaks below S1 + 1d trend down + volume spike
            elif price_now < s1 and price_now < trend_1d and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to pivot or 1d trend turns down
            if price_now < pivot or price_now < trend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to pivot or 1d trend turns up
            if price_now > pivot or price_now > trend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0