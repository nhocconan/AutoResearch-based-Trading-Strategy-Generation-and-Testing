#!/usr/bin/env python3
# 6h_weekly_pivot_volume_confirmation_v3
# Hypothesis: 6h strategy using weekly pivot points (R1,R2,S1,S2) for breakout/breakdown entries with volume confirmation (>1.8x 24-bar avg volume). Uses 1d HTF EMA(50) for trend alignment to avoid counter-trend trades. Discrete position sizing (0.25) to minimize fee churn. Target: 12-37 trades/year (50-150 total over 4 years). Weekly pivots provide structural support/resistance that works in both bull/bear markets; volume confirms conviction; HTF EMA ensures alignment with higher timeframe trend to avoid whipsaws.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_volume_confirmation_v3"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly pivot points from prior week (using weekly OHLC)
    df_1w = get_htf_data(prices, '1w')
    # Calculate weekly pivot points: P = (H+L+C)/3, R1 = 2P-L, S1 = 2P-H, R2 = P+(H-L), S2 = P-(H-L)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    pivot_point = (weekly_high + weekly_low + weekly_close) / 3
    weekly_range = weekly_high - weekly_low
    r1 = 2 * pivot_point - weekly_low
    s1 = 2 * pivot_point - weekly_high
    r2 = pivot_point + weekly_range
    s2 = pivot_point - weekly_range
    # Align weekly pivot levels to 6h timeframe (completed weekly bar only)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # Volume average for confirmation (24-period = 4 days of 6h bars)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=24, min_periods=24).mean().values
    
    # Multi-timeframe: 1d EMA(50) trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    close_1d_s = pd.Series(close_1d)
    ema_50_1d = close_1d_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(volume[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.8x 24-period average
        volume_confirmed = volume[i] > 1.8 * volume_ma[i]
        # HTF trend filter: price above/below 1d EMA(50)
        htf_uptrend = close[i] > ema_50_1d_aligned[i]
        htf_downtrend = close[i] < ema_50_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price closes below S1 (weekly support 1)
            if close[i] < s1_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above R1 (weekly resistance 1)
            if close[i] > r1_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for breakout above R1 with volume and HTF uptrend
            bullish_breakout = (close[i] > r1_aligned[i-1]) and volume_confirmed and htf_uptrend
            # Check for breakdown below S1 with volume and HTF downtrend
            bearish_breakdown = (close[i] < s1_aligned[i-1]) and volume_confirmed and htf_downtrend
            
            if bullish_breakout:
                position = 1
                signals[i] = 0.25
            elif bearish_breakdown:
                position = -1
                signals[i] = -0.25
    
    return signals