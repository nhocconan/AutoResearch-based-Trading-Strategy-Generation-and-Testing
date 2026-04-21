#!/usr/bin/env python3
"""
6h Weekly Pivot R2/S2 Breakout with Volume Confirmation and ATR Stop
Hypothesis: Weekly pivot points R2/S2 act as significant support/resistance levels.
Breakouts with volume confirmation capture institutional moves, while the 6h timeframe
reduces noise compared to lower timeframes. ATR-based stops limit losses during false breakouts.
Designed for low trade frequency (12-37/year) to minimize fee drag and improve generalization.
Works in both bull and bear markets by capturing breakouts in either direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
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
    
    # Calculate weekly pivot points: P = (H+L+C)/3, R2 = P + (H-L), S2 = P - (H-L)
    pivot_weekly = (high_weekly + low_weekly + close_weekly) / 3.0
    r2_weekly = pivot_weekly + (high_weekly - low_weekly)
    s2_weekly = pivot_weekly - (high_weekly - low_weekly)
    
    # Align weekly indicators to 6h timeframe
    atr_weekly_aligned = align_htf_to_ltf(prices, df_weekly, atr_weekly)
    r2_weekly_aligned = align_htf_to_ltf(prices, df_weekly, r2_weekly)
    s2_weekly_aligned = align_htf_to_ltf(prices, df_weekly, s2_weekly)
    
    # Main timeframe data (6h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(atr_weekly_aligned[i]) or np.isnan(r2_weekly_aligned[i]) or np.isnan(s2_weekly_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        atr = atr_weekly_aligned[i]
        r2 = r2_weekly_aligned[i]
        s2 = s2_weekly_aligned[i]
        vol_current = volume[i]
        
        # Volume filter: current volume > 2.0x 30-period average
        vol_ma = np.mean(volume[max(0, i-30):i]) if i >= 30 else volume[i]
        vol_ok = vol_current > 2.0 * vol_ma
        
        if position == 0:
            # Long breakout: price breaks above R2 with volume confirmation
            if price > r2 and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below S2 with volume confirmation
            elif price < s2 and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S2 (failed breakout) or ATR-based stop
            if price < s2 or (i > 0 and close[i-1] > s2 and price < close[i-1] - 1.5 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R2 (failed breakdown) or ATR-based stop
            if price > r2 or (i > 0 and close[i-1] < r2 and price > close[i-1] + 1.5 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_R2S2_Breakout_Volume_ATRFilter"
timeframe = "6h"
leverage = 1.0