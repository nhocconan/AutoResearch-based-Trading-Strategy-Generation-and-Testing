#!/usr/bin/env python3
"""
6h_WeeklyPivot_R3S3_Reversal_S4S4_Breakout
Hypothesis: On 6h timeframe, weekly pivot levels act as strong support/resistance.
- Mean reversion: Price touching weekly R3/S3 with rejection (close back inside R2/S2) signals reversal.
- Breakout: Price breaking weekly S4/R4 with volume and trend continuation signals strong momentum.
Works in both bull and bear markets by combining mean reversion at extremes and breakout continuation.
Uses weekly pivots calculated from prior week's OHLC to avoid look-ahead.
Target: 20-40 trades/year on 6h (~80-160 total over 4 years) to minimize fee drag.
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
    
    # Get weekly data for pivot calculation (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points using prior week's data
    # P = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    # R4 = R3 + (H - L), S4 = S3 - (H - L)
    high_1w = df_1w['high']
    low_1w = df_1w['low']
    close_1w = df_1w['close']
    
    pivot = (high_1w + low_1w + close_1w) / 3
    r1 = 2 * pivot - low_1w
    s1 = 2 * pivot - high_1w
    r2 = pivot + (high_1w - low_1w)
    s2 = pivot - (high_1w - low_1w)
    r3 = high_1w + 2 * (pivot - low_1w)
    s3 = low_1w - 2 * (high_1w - pivot)
    r4 = r3 + (high_1w - low_1w)
    s4 = s3 - (high_1w - low_1w)
    
    # Shift by 1 to use previous week's levels only (avoid look-ahead)
    r1_prev = r1.shift(1).values
    s1_prev = s1.shift(1).values
    r2_prev = r2.shift(1).values
    s2_prev = s2.shift(1).values
    r3_prev = r3.shift(1).values
    s3_prev = s3.shift(1).values
    r4_prev = r4.shift(1).values
    s4_prev = s4.shift(1).values
    
    # Align to 6h timeframe
    r1_a = align_htf_to_ltf(prices, df_1w, r1_prev)
    s1_a = align_htf_to_ltf(prices, df_1w, s1_prev)
    r2_a = align_htf_to_ltf(prices, df_1w, r2_prev)
    s2_a = align_htf_to_ltf(prices, df_1w, s2_prev)
    r3_a = align_htf_to_ltf(prices, df_1w, r3_prev)
    s3_a = align_htf_to_ltf(prices, df_1w, s3_prev)
    r4_a = align_htf_to_ltf(prices, df_1w, r4_prev)
    s4_a = align_htf_to_ltf(prices, df_1w, s4_prev)
    
    # 60-period EMA for trend filter (on 6h close)
    close_series = pd.Series(close)
    ema60 = close_series.ewm(span=60, adjust=False, min_periods=60).mean().values
    
    # Volume spike: 1.5x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 60  # need EMA60
    
    for i in range(start_idx, n):
        # Skip if any weekly pivot level is NaN
        if (np.isnan(r1_a[i]) or np.isnan(s1_a[i]) or
            np.isnan(r2_a[i]) or np.isnan(s2_a[i]) or
            np.isnan(r3_a[i]) or np.isnan(s3_a[i]) or
            np.isnan(r4_a[i]) or np.isnan(s4_a[i]) or
            np.isnan(ema60[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_spike = volume_spike[i]
        ema_trend = ema60[i]
        
        if position == 0:
            # Mean reversion long: price touches S3 but closes back above S2
            if price <= s3_a[i] and close[i] > s2_a[i]:
                signals[i] = 0.25
                position = 1
            # Mean reversion short: price touches R3 but closes back below R2
            elif price >= r3_a[i] and close[i] < r2_a[i]:
                signals[i] = -0.25
                position = -1
            # Breakout long: price breaks above R4 with volume and uptrend
            elif price > r4_a[i] and vol_spike and price > ema_trend:
                signals[i] = 0.25
                position = 1
            # Breakout short: price breaks below S4 with volume and downtrend
            elif price < s4_a[i] and vol_spike and price < ema_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position
            signals[i] = 0.25
            # Exit: mean reversion at R1 or break below S2 (failed breakout)
            if price >= r1_a[i] or price < s2_a[i]:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position
            signals[i] = -0.25
            # Exit: mean reversion at S1 or break above R2 (failed breakdown)
            if price <= s1_a[i] or price > r2_a[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WeeklyPivot_R3S3_Reversal_S4S4_Breakout"
timeframe = "6h"
leverage = 1.0