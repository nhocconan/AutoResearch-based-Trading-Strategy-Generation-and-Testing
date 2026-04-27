#!/usr/bin/env python3
"""
4h Camarilla Pivot Breakout with 12h Volume Spike and 1d Trend Filter.
Long when price breaks above R1 with 12h volume spike and 1d EMA50 up.
Short when price breaks below S1 with 12h volume spike and 1d EMA50 down.
Exit when price returns to Pivot point or trend changes.
Designed for low frequency (20-40 trades/year) to minimize fee drag.
Uses Camarilla levels from daily data and 12h volume confirmation.
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
    
    # Get daily data for Camarilla pivot and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 12h data for volume spike filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Using previous day's data to avoid look-ahead
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla formula: Range = High - Low
    range_prev = prev_high - prev_low
    
    # Calculate levels (using previous day's close as base)
    R1 = prev_close + range_prev * 1.1 / 12
    S1 = prev_close - range_prev * 1.1 / 12
    Pivot = prev_close
    
    # Align Camarilla levels to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    Pivot_aligned = align_htf_to_ltf(prices, df_1d, Pivot)
    
    # Calculate 12h volume moving average for spike detection
    vol_12h = df_12h['volume'].values
    vol_ma_20_12h = np.full_like(vol_12h, np.nan, dtype=np.float64)
    for i in range(19, len(vol_12h)):
        vol_ma_20_12h[i] = np.mean(vol_12h[i-19:i+1])
    
    # Align 12h volume MA to 4h timeframe
    vol_ma_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Camarilla (1 day lag) + volume MA (20) + EMA (50)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(Pivot_aligned[i]) or np.isnan(vol_ma_aligned[i]) or
            np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        
        # Current levels
        R1_level = R1_aligned[i]
        S1_level = S1_aligned[i]
        Pivot_level = Pivot_aligned[i]
        vol_ma_level = vol_ma_aligned[i]
        ema_trend = ema_50_aligned[i]
        
        # Volume filter: volume > 1.5x 12h average volume (adjusted for timeframe)
        # Scale factor: 12h has 48x 4h bars, so average 4h volume = 12h volume / 48
        vol_filter = vol_now > 1.5 * (vol_ma_level / 48.0)
        
        if position == 0:
            # Bull: price breaks above R1 + volume spike + EMA50 trending up
            if price_now > R1_level and vol_filter and price_now > ema_trend:
                signals[i] = size
                position = 1
            # Bear: price breaks below S1 + volume spike + EMA50 trending down
            elif price_now < S1_level and vol_filter and price_now < ema_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to Pivot or trend turns down
            if price_now < Pivot_level or price_now < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to Pivot or trend turns up
            if price_now > Pivot_level or price_now > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_12hVolume_1dEMA50"
timeframe = "4h"
leverage = 1.0