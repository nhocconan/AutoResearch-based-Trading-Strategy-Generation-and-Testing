#!/usr/bin/env python3
"""
1d_1w_12h_Camarilla_Pivot_Breakout_Volume
Hypothesis: Use weekly Camarilla pivot levels for structural support/resistance, enter on breakout above R4 or below S4 with volume confirmation and 12h EMA trend filter. Designed for low frequency (target 15-25 trades/year) to minimize fee drag. Works in bull markets via upside breakouts and bear markets via downside breakdowns. Uses 12h EMA34 to filter direction and avoid counter-trend trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for Camarilla pivots
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Camarilla pivot levels (based on previous week)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Pivot point
    pivot = (high_1w + low_1w + close_1w) / 3.0
    # Ranges
    range_hl = high_1w - low_1w
    # Camarilla levels
    r4 = pivot + (range_hl * 1.1 / 2)
    r3 = pivot + (range_hl * 1.1 / 4)
    r2 = pivot + (range_hl * 1.1 / 6)
    r1 = pivot + (range_hl * 1.1 / 12)
    s1 = pivot - (range_hl * 1.1 / 12)
    s2 = pivot - (range_hl * 1.1 / 6)
    s3 = pivot - (range_hl * 1.1 / 4)
    s4 = pivot - (range_hl * 1.1 / 2)
    
    # Align weekly levels to daily (using previous week's levels)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # EMA34 on 12h
    ema34_12h = np.full_like(close_12h, np.nan)
    if len(close_12h) >= 34:
        ema = np.zeros(len(close_12h))
        ema[0] = close_12h[0]
        alpha = 2.0 / (34 + 1)
        for i in range(1, len(close_12h)):
            ema[i] = alpha * close_12h[i] + (1 - alpha) * ema[i-1]
        ema34_12h = ema
    
    # Align EMA34 to daily
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Volume confirmation: volume > 1.5x 20-day average
    volume = prices['volume'].values
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        for i in range(20, len(volume)):
            vol_ma[i] = np.mean(volume[i-20:i])
    
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # volume MA needs 20 days
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema34_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: break above R4 with volume and EMA34 > price (uptrend)
            if close[i] > r4_aligned[i] and vol_confirm and ema34_12h_aligned[i] > close[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below S4 with volume and EMA34 < price (downtrend)
            elif close[i] < s4_aligned[i] and vol_confirm and ema34_12h_aligned[i] < close[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: close below R3 or EMA34 turns down
            if close[i] < r3_aligned[i] or (i > 0 and not np.isnan(ema34_12h_aligned[i-1]) and ema34_12h_aligned[i] < ema34_12h_aligned[i-1]):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above S3 or EMA34 turns up
            if close[i] > s3_aligned[i] or (i > 0 and not np.isnan(ema34_12h_aligned[i-1]) and ema34_12h_aligned[i] > ema34_12h_aligned[i-1]):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_12h_Camarilla_Pivot_Breakout_Volume"
timeframe = "1d"
leverage = 1.0