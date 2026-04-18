#!/usr/bin/env python3
"""
6h_WeeklyPivot_Camberilla_Breakout_Filter
Hypothesis: Trade breakouts from weekly Camarilla pivot levels (R4/S4) with 1d volume confirmation and 1d trend filter (EMA50). In bull markets, buy R4 breakouts; in bear markets, sell S4 breakdowns. Weekly pivots provide strong institutional levels; volume confirms institutional participation; EMA50 filter ensures trading with higher timeframe trend. Designed for low frequency (10-30 trades/year) to minimize fee impact on 6H timeframe.
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
    
    # Get weekly data for Camarilla pivots
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly Camarilla levels
    # Formula: R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4), etc.
    # Where C = (H+L+C)/3 (typical price)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Typical price
    weekly_tp = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Camarilla levels
    r4 = weekly_tp + ((weekly_high - weekly_low) * 1.1 / 2.0)
    r3 = weekly_tp + ((weekly_high - weekly_low) * 1.1 / 4.0)
    s3 = weekly_tp - ((weekly_high - weekly_low) * 1.1 / 4.0)
    s4 = weekly_tp - ((weekly_high - weekly_low) * 1.1 / 2.0)
    
    # Align weekly Camarilla levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    
    # Get 1d data for volume and trend filters
    df_1d = get_htf_data(prices, '1d')
    
    # 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = np.full_like(vol_1d, np.nan)
    if len(vol_1d) >= 20:
        for i in range(20, len(vol_1d)):
            vol_ma_1d[i] = np.mean(vol_1d[i-20:i])
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (close_1d[i] * 2 + ema_50_1d[i-1] * 49) / 50
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # EMA50 needs 50, vol MA needs 20
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x 1d average volume
        # Convert 1d average to equivalent 6h period (1d = 4x 6h)
        vol_threshold = vol_ma_1d_aligned[i] * 1.5 / 4.0
        vol_confirm = volume[i] > vol_threshold
        
        if position == 0:
            # Long: price breaks above R4 with volume and above 1d EMA50 (uptrend)
            if close[i] > r4_aligned[i] and vol_confirm and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S4 with volume and below 1d EMA50 (downtrend)
            elif close[i] < s4_aligned[i] and vol_confirm and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below R3 or volume dries up
            if close[i] < r3_aligned[i] or volume[i] < vol_threshold * 0.5:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above S3 or volume dries up
            if close[i] > s3_aligned[i] or volume[i] < vol_threshold * 0.5:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_Camberilla_Breakout_Filter"
timeframe = "6h"
leverage = 1.0