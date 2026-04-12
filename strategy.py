#!/usr/bin/env python3
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
    
    # Get daily data for pivot points and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 14-day ATR on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = np.full(len(close_1d), np.nan)
    for i in range(14, len(atr_1d)):
        atr_1d[i] = np.nanmean(tr[i-13:i+1])
    
    # Align ATR to 6h timeframe
    atr_6h = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate daily pivot points using previous day's data
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # Key levels: R3, S3, R4, S4 (Camarilla)
    r3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    s3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    r4 = close_1d + (high_1d - low_1d) * 1.1 / 2
    s4 = close_1d - (high_1d - low_1d) * 1.1 / 2
    
    # Align levels to 6h timeframe
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate volume moving average (20-period on 6h)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(atr_6h[i]) or 
            np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or 
            np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.8x 20-period average
        volume_filter = volume[i] > vol_ma[i] * 1.8
        
        # Entry conditions: R4/S4 breakout with volume confirmation (trend continuation)
        long_breakout = (close[i] > r4_6h[i]) and volume_filter
        short_breakout = (close[i] < s4_6h[i]) and volume_filter
        
        # Entry conditions: R3/S3 fade with volume confirmation (mean reversion)
        long_fade = (close[i] < s3_6h[i]) and volume_filter
        short_fade = (close[i] > r3_6h[i]) and volume_filter
        
        # Exit conditions: touch opposite level or ATR stop
        long_exit = (close[i] < s3_6h[i]) or (close[i] < close[i-1] - 2.0 * atr_6h[i])
        short_exit = (close[i] > r3_6h[i]) or (close[i] > close[i-1] + 2.0 * atr_6h[i])
        
        # Prioritize fade signals in ranging markets, breakout in trending
        if long_fade and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_fade and position != -1:
            position = -1
            signals[i] = -0.25
        elif long_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_camarilla_r3s3_r4s4_volume_filter_v1"
timeframe = "6h"
leverage = 1.0