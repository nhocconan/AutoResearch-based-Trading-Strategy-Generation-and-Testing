#!/usr/bin/env python3
"""
6h Weekly Pivot Direction + Volume Confirmation
Hypothesis: Weekly pivot levels provide strong support/resistance in both bull and bear markets.
Price breaking above weekly R1 with volume and above weekly EMA50 indicates bullish momentum.
Price breaking below weekly S1 with volume and below weekly EMA50 indicates bearish momentum.
Weekly trend filter avoids counter-trend trades. Designed for low frequency (12-30 trades/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot and trend filter (once before loop)
    df_w = get_htf_data(prices, '1w')
    
    # Calculate weekly high, low, close for pivot levels
    weekly_high = df_w['high'].values
    weekly_low = df_w['low'].values
    weekly_close = df_w['close'].values
    
    # Calculate weekly pivot: P = (H+L+C)/3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_range = weekly_high - weekly_low
    # Weekly R1 = P + (H-L) = P + range
    weekly_r1 = weekly_pivot + weekly_range
    # Weekly S1 = P - (H-L) = P - range
    weekly_s1 = weekly_pivot - weekly_range
    
    # Calculate weekly EMA50 for trend filter
    ema_50_w = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly levels to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_w, weekly_r1)
    s1_aligned = align_htf_to_ltf(prices, df_w, weekly_s1)
    pivot_aligned = align_htf_to_ltf(prices, df_w, weekly_pivot)
    ema_50_w_aligned = align_htf_to_ltf(prices, df_w, ema_50_w)
    
    # Volume spike detection (1.5x 30-period average)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 100  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or
            np.isnan(pivot_aligned[i]) or
            np.isnan(ema_50_w_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1_level = r1_aligned[i]
        s1_level = s1_aligned[i]
        pivot_level = pivot_aligned[i]
        ema_50 = ema_50_w_aligned[i]
        
        if position == 0:
            # Long: break above weekly R1 with volume spike and above weekly EMA50
            if (price > r1_level and volume_spike[i] and price > ema_50):
                signals[i] = 0.25
                position = 1
            # Short: break below weekly S1 with volume spike and below weekly EMA50
            elif (price < s1_level and volume_spike[i] and price < ema_50):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position management
            signals[i] = 0.25
            # Exit: price returns to weekly pivot or below weekly EMA50 (trend change)
            if price <= pivot_level or price < ema_50:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.25
            # Exit: price returns to weekly pivot or above weekly EMA50 (trend change)
            if price >= pivot_level or price > ema_50:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WeeklyPivot_R1S1_Breakout_VolumeSpike_1wEMA50"
timeframe = "6h"
leverage = 1.0