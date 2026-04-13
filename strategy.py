#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 12h trend filter and volume confirmation.
# Camarilla levels from 1d: R3/S3 for mean reversion, R4/S4 for breakout continuation.
# Combined with 12h EMA trend filter and volume spikes to confirm breakouts.
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each 1d bar
    camarilla_r4 = np.zeros(len(close_1d))
    camarilla_r3 = np.zeros(len(close_1d))
    camarilla_s3 = np.zeros(len(close_1d))
    camarilla_s4 = np.zeros(len(close_1d))
    pivot = np.zeros(len(close_1d))
    
    for i in range(len(close_1d)):
        high_i = high_1d[i]
        low_i = low_1d[i]
        close_i = close_1d[i]
        range_i = high_i - low_i
        
        pivot[i] = (high_i + low_i + close_i) / 3
        camarilla_r4[i] = close_i + range_i * 1.1 / 2
        camarilla_r3[i] = close_i + range_i * 1.1 / 4
        camarilla_s3[i] = close_i - range_i * 1.1 / 4
        camarilla_s4[i] = close_i - range_i * 1.1 / 2
    
    # Align 1d Camarilla levels to 6h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # 12-hour data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # EMA(20) for 12h trend filter
    ema20_12h = np.zeros(len(close_12h))
    ema_multiplier = 2 / (20 + 1)
    ema20_12h[0] = close_12h[0]
    for i in range(1, len(close_12h)):
        ema20_12h[i] = (close_12h[i] - ema20_12h[i-1]) * ema_multiplier + ema20_12h[i-1]
    
    # Align 12h EMA to 6h timeframe
    ema20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema20_12h)
    
    # Average volume (10-period = 10*6h = 60h ~ 2.5 days) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(10, n):
        avg_volume[i] = np.mean(volume[i-10:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(10, n):
        # Skip if any required data is not ready
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(ema20_12h_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_trend = ema20_12h_aligned[i]
        r4 = camarilla_r4_aligned[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        s4 = camarilla_s4_aligned[i]
        
        # Volume confirmation: current volume > 1.3x average volume
        volume_confirm = vol > 1.3 * avg_vol
        
        if position == 0:
            # Long breakout: price > R4 with volume confirmation and above 12h EMA
            if (price > r4 and volume_confirm and price > ema_trend):
                position = 1
                signals[i] = position_size
            # Short breakdown: price < S4 with volume confirmation and below 12h EMA
            elif (price < s4 and volume_confirm and price < ema_trend):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below R3 or below 12h EMA
            if (price < r3 or price < ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above S3 or above 12h EMA
            if (price > s3 or price > ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_12h_Camarilla_Breakout_Trend_Volume"
timeframe = "6h"
leverage = 1.0