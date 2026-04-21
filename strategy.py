#!/usr/bin/env python3
"""
6h Weekly Pivot R2/S2 Breakout with Volume and Momentum Confirmation
Hypothesis: Weekly pivot points R2/S2 act as significant institutional support/resistance.
Breakouts above R2 or below S2 with volume and momentum capture sustained moves in both bull and bear markets.
Volume confirms institutional participation, momentum filters out weak breakouts. Designed for low trade frequency
(~12-37/year) to minimize fee drag. Uses 6h timeframe for balance between signal quality and trade frequency.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data once for pivot points
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly pivot points: P = (H+L+C)/3, R2 = P + (H-L), S2 = P - (H-L)
    pivot_weekly = (high_weekly + low_weekly + close_weekly) / 3.0
    r2_weekly = pivot_weekly + (high_weekly - low_weekly)
    s2_weekly = pivot_weekly - (high_weekly - low_weekly)
    
    # Align weekly pivot points to 6h timeframe
    r2_weekly_aligned = align_htf_to_ltf(prices, df_weekly, r2_weekly)
    s2_weekly_aligned = align_htf_to_ltf(prices, df_weekly, s2_weekly)
    
    # Main timeframe data (6h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Momentum: 12-period ROC (rate of change)
    roc = np.zeros_like(close)
    for i in range(12, n):
        roc[i] = (close[i] - close[i-12]) / close[i-12] * 100
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(r2_weekly_aligned[i]) or np.isnan(s2_weekly_aligned[i]) or np.isnan(roc[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        r2 = r2_weekly_aligned[i]
        s2 = s2_weekly_aligned[i]
        vol_current = volume[i]
        
        # Volume filter: current volume > 2.0x 24-period average (4 days)
        vol_ma = np.mean(volume[max(0, i-24):i]) if i >= 24 else volume[i]
        vol_ok = vol_current > 2.0 * vol_ma
        
        # Momentum filter: ROC > 0 for long, ROC < 0 for short
        mom_long = roc[i] > 0
        mom_short = roc[i] < 0
        
        if position == 0:
            # Long breakout: price breaks above R2 with volume and momentum confirmation
            if price > r2 and vol_ok and mom_long:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below S2 with volume and momentum confirmation
            elif price < s2 and vol_ok and mom_short:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S2 (failed breakout) or momentum reversal
            if price < s2 or roc[i] < -1.0:  # momentum turns negative
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R2 (failed breakdown) or momentum reversal
            if price > r2 or roc[i] > 1.0:  # momentum turns positive
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_R2S2_Breakout_Volume_Momentum"
timeframe = "6h"
leverage = 1.0