#!/usr/bin/env python3
"""
6h Pivot Reversal with Volume Confirmation and 12h Trend Filter
Hypothesis: Prices often reverse at daily pivot levels (S1/R1, S2/R2) with volume confirmation.
We use 12h EMA trend filter to avoid counter-trend trades and enter on 6h bounces from
support/resistance levels. This strategy targets 15-25 trades/year to minimize fee drag
while capturing high-probability reversals in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot points (once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate daily pivot points: P = (H+L+C)/3, S1 = 2P-H, R1 = 2P-L
    # S2 = P-(H-L), R2 = P+(H-L)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    s1 = 2 * pivot - high_1d
    r1 = 2 * pivot - low_1d
    s2 = pivot - (high_1d - low_1d)
    r2 = pivot + (high_1d - low_1d)
    
    # Align pivot levels to 6h timeframe (1 bar delay for completed daily bar)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    
    # Get 12h EMA34 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    ema34_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Volume filter: current volume > 1.3x 20-period volume average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i]) or
            np.isnan(s2_aligned[i]) or np.isnan(r2_aligned[i]) or np.isnan(ema34_12h_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ok = vol_filter[i]
        trend = ema34_12h_aligned[i]
        
        if position == 0:
            # Long: price near S1/S2 with volume, in uptrend
            if vol_ok and price > trend:
                # Near S1 (within 0.5% tolerance) or S2
                if abs(price - s1_aligned[i]) / s1_aligned[i] < 0.005 or abs(price - s2_aligned[i]) / s2_aligned[i] < 0.005:
                    signals[i] = 0.25
                    position = 1
            # Short: price near R1/R2 with volume, in downtrend
            elif vol_ok and price < trend:
                # Near R1 (within 0.5% tolerance) or R2
                if abs(price - r1_aligned[i]) / r1_aligned[i] < 0.005 or abs(price - r2_aligned[i]) / r2_aligned[i] < 0.005:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit if price reaches pivot or trend weakens
            if price >= pivot_aligned[i] or price < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit if price reaches pivot or trend weakens
            if price <= pivot_aligned[i] or price > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Pivot_Reversal_Volume_12hTrend"
timeframe = "6h"
leverage = 1.0