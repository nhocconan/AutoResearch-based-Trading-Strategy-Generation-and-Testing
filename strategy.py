#!/usr/bin/env python3
"""
1d_Pivot_Reversion_v1
Hypothesis: Price tends to revert to weekly pivot levels (resistance/support) with high probability.
In both bull and bear markets, price often retraces to key pivot levels before continuing the trend.
Uses weekly pivot points calculated from weekly OHLC, with 1d close crossing as entry signal.
Filters: price must be trending (using 20-period EMA on 1d) and volume above average.
Position size: 0.25 for clear reversion signals.
Target: 20-60 trades over 4 years (5-15/year) on 1d timeframe.
"""

name = "1d_Pivot_Reversion_v1"
timeframe = "1d"
leverage = 1.0

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
    
    # === Weekly Data for Pivot Points ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points: P = (H + L + C)/3
    # Resistance: R1 = 2*P - L, S1 = 2*P - H
    # Extended: R2 = P + (H - L), S2 = P - (H - L)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    r2 = pivot + (weekly_high - weekly_low)
    s2 = pivot - (weekly_high - weekly_low)
    
    # Align weekly pivot levels to daily timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # === 1d Indicators for Filtering ===
    # 20-period EMA for trend filter
    close_series = pd.Series(close)
    ema20 = close_series.ewm(span=20, adjust=False, min_periods=20).values
    
    # Volume average (20-period) for volume filter
    volume_series = pd.Series(volume)
    vol_avg = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(ema20[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long setup: price near support levels in uptrend with volume confirmation
            near_s1 = abs(close[i] - s1_aligned[i]) / s1_aligned[i] < 0.005  # within 0.5%
            near_s2 = abs(close[i] - s2_aligned[i]) / s2_aligned[i] < 0.005  # within 0.5%
            in_uptrend = close[i] > ema20[i]
            vol_confirm = volume[i] > vol_avg[i]
            
            if (near_s1 or near_s2) and in_uptrend and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short setup: price near resistance levels in downtrend with volume confirmation
            elif (abs(close[i] - r1_aligned[i]) / r1_aligned[i] < 0.005 or 
                  abs(close[i] - r2_aligned[i]) / r2_aligned[i] < 0.005) and \
                 close[i] < ema20[i] and volume[i] > vol_avg[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reaches pivot or shows weakness
            if close[i] >= pivot_aligned[i] or close[i] < ema20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price reaches pivot or shows strength
            if close[i] <= pivot_aligned[i] or close[i] > ema20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals