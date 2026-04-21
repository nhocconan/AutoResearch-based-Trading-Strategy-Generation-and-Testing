#!/usr/bin/env python3
"""
1d Weekly Pivot R1/S1 Breakout with Volume Confirmation and ATR Stop
Hypothesis: Weekly pivot points R1/S1 act as strong support/resistance levels.
Breakouts with volume confirmation capture institutional moves in both bull and bear markets.
ATR-based stops limit losses during false breakouts. Designed for low trade frequency
(~7-25/year) to minimize fee drag and improve generalization. Uses 1d timeframe to capture daily momentum.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load weekly data once for pivot points and ATR
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly True Range for ATR
    tr1 = np.abs(high_weekly - low_weekly)
    tr2 = np.abs(high_weekly - np.roll(close_weekly, 1))
    tr3 = np.abs(low_weekly - np.roll(close_weekly, 1))
    tr1[0] = high_weekly[0] - low_weekly[0]
    tr2[0] = np.abs(high_weekly[0] - close_weekly[0])
    tr3[0] = np.abs(low_weekly[0] - close_weekly[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_weekly = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate weekly pivot points: P = (H+L+C)/3, R1 = 2*P - L, S1 = 2*P - H
    pivot_weekly = (high_weekly + low_weekly + close_weekly) / 3.0
    r1_weekly = 2 * pivot_weekly - low_weekly
    s1_weekly = 2 * pivot_weekly - high_weekly
    
    # Align weekly indicators to 1d timeframe
    atr_weekly_aligned = align_htf_to_ltf(prices, df_weekly, atr_weekly)
    r1_weekly_aligned = align_htf_to_ltf(prices, df_weekly, r1_weekly)
    s1_weekly_aligned = align_htf_to_ltf(prices, df_weekly, s1_weekly)
    
    # Main timeframe data (1d)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if NaN in critical values
        if (np.isnan(atr_weekly_aligned[i]) or np.isnan(r1_weekly_aligned[i]) or np.isnan(s1_weekly_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        atr = atr_weekly_aligned[i]
        r1 = r1_weekly_aligned[i]
        s1 = s1_weekly_aligned[i]
        vol_current = volume[i]
        
        # Volume filter: current volume > 2.0x 30-period average
        vol_ma = np.mean(volume[max(0, i-30):i]) if i >= 30 else volume[i]
        vol_ok = vol_current > 2.0 * vol_ma
        
        if position == 0:
            # Long breakout: price breaks above R1 with volume confirmation
            if price > r1 and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below S1 with volume confirmation
            elif price < s1 and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S1 (failed breakout) or ATR-based stop
            if price < s1 or (i > 0 and close[i-1] > s1 and price < close[i-1] - 1.5 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1 (failed breakdown) or ATR-based stop
            if price > r1 or (i > 0 and close[i-1] < r1 and price > close[i-1] + 1.5 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyPivot_R1S1_Breakout_Volume_ATRFilter"
timeframe = "1d"
leverage = 1.0