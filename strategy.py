#!/usr/bin/env python3
"""
6h Weekly Pivot Range Fade with Volume Confirmation
Fades price at weekly pivot extremes (S4/R4) when price reaches these levels with volume confirmation.
Uses daily trend filter to avoid fading against strong daily momentum.
Designed for low trade frequency with mean-reversion edge at extreme weekly levels.
Works in ranging markets and provides counter-trend entries during overextended moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (standard floor trader pivots)
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    # Weekly S4 and R4 levels (most extreme)
    s4_1w = pivot_1w - 3 * range_1w
    r4_1w = pivot_1w + 3 * range_1w
    
    # Align weekly levels to 6h timeframe (need extra delay for weekly confirmation)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4_1w, additional_delay_bars=1)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4_1w, additional_delay_bars=1)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation (1.5x 6-period average)
    vol_ma = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    volume_confirmed = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(s4_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        s4_val = s4_aligned[i]
        r4_val = r4_aligned[i]
        ema_trend = ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: price at weekly S4 with volume confirmation and above daily EMA (bullish bias)
            if (price <= s4_val and 
                volume_confirmed[i] and 
                price > ema_trend):
                signals[i] = 0.25
                position = 1
            # Short: price at weekly R4 with volume confirmation and below daily EMA (bearish bias)
            elif (price >= r4_val and 
                  volume_confirmed[i] and 
                  price < ema_trend):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position: hold until mean reversion to weekly pivot
            signals[i] = 0.25
            # Exit when price returns to weekly pivot level
            if price >= pivot_1w[-1] if len(pivot_1w) > 0 else pivot_1w[0]:  # Simplified exit
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position: hold until mean reversion to weekly pivot
            signals[i] = -0.25
            # Exit when price returns to weekly pivot level
            if price <= pivot_1w[-1] if len(pivot_1w) > 0 else pivot_1w[0]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WeeklyPivot_RangeFade_Volume_DailyTrend"
timeframe = "6h"
leverage = 1.0