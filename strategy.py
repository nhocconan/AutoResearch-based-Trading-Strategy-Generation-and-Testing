#!/usr/bin/env python3
"""
6h_camarilla_pivot_1d_ema_volume_v3
Hypothesis: On 6-hour timeframe, use daily Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) with EMA200 trend filter and volume confirmation. Fade at R3/S3 in ranging markets, breakout continuation at R4/S4 in trending markets. Designed for low frequency (12-37 trades/year) to avoid fee drag while capturing both mean reversion and trend continuation. Works in bull/bear by using daily EMA200 trend filter to determine regime.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_1d_ema_volume_v3"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    
    # Calculate daily EMA200 for trend filter
    d_ema200 = pd.Series(d_close).ewm(span=200, adjust=False).mean().values
    d_ema200_aligned = align_htf_to_ltf(prices, df_1d, d_ema200)
    
    # Calculate daily Camarilla pivot levels (using previous day's data)
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R4 = C + (H-L) * 1.500
    # R3 = C + (H-L) * 1.250
    # S3 = C - (H-L) * 1.250
    # S4 = C - (H-L) * 1.500
    pivot = (d_high + d_low + d_close) / 3.0
    rang = d_high - d_low
    r4 = d_close + rang * 1.500
    r3 = d_close + rang * 1.250
    s3 = d_close - rang * 1.250
    s4 = d_close - rang * 1.500
    
    # Align pivot levels to 6h timeframe (use previous day's levels)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate 24-period average volume for confirmation (4 days)
    vol_avg = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(24, n):  # Start after volume average warmup
        # Skip if daily EMA200 not available
        if np.isnan(d_ema200_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Determine trend based on price vs daily EMA200
        uptrend = close[i] > d_ema200_aligned[i]
        downtrend = close[i] < d_ema200_aligned[i]
        
        # Volume confirmation: current volume > 1.3x 24-period average
        vol_confirm = volume[i] > 1.3 * vol_avg[i] if not np.isnan(vol_avg[i]) else False
        
        if position == 1:  # Long position
            # Exit when price reaches S3 (mean reversion target) or breaks above R4 (trailing stop)
            if close[i] <= s3_aligned[i] or close[i] >= r4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when price reaches R3 (mean reversion target) or breaks below S4 (trailing stop)
            if close[i] >= r3_aligned[i] or close[i] <= s4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Mean reversion entries at R3/S3 (fade extreme levels)
            mean_reversion_long = (close[i] <= s3_aligned[i]) and downtrend and vol_confirm
            mean_reversion_short = (close[i] >= r3_aligned[i]) and uptrend and vol_confirm
            
            # Breakout entries at R4/S4 (continuation in trend direction)
            breakout_long = (close[i] >= r4_aligned[i]) and uptrend and vol_confirm
            breakout_short = (close[i] <= s4_aligned[i]) and downtrend and vol_confirm
            
            if mean_reversion_long or breakout_long:
                position = 1
                signals[i] = 0.25
            elif mean_reversion_short or breakout_short:
                position = -1
                signals[i] = -0.25
    
    return signals