#!/usr/bin/env python3
# 4h_camarilla_pivot_volume_chop_v1
# Hypothesis: 4h strategy using daily Camarilla pivot levels (from 1d HTF) for mean-reversion entries, volume confirmation (>1.5x 20-bar avg volume), and chop regime filter (CHOP(14) between 38.2 and 61.8). Enters long when price touches S3 with volume and chop>38.2; enters short when price touches R3 with volume and chop<61.8. Exits at opposite pivot level (S1/R1) or close beyond S4/R4. Uses discrete sizing (0.25) to limit fee churn. Target: 20-50 trades/year (80-200 total over 4 years). Daily Camarilla pivots provide intraday support/resistance that works in ranging markets; volume confirms institutional interest; chop filter avoids trending markets where mean reversion fails.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_camarilla_pivot_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period = ~3.3 days of 4h bars)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Multi-timeframe: daily Camarilla pivot levels (from 1d HTF)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r3 = pivot + (range_1d * 1.1 / 4)
    r4 = pivot + (range_1d * 1.1 / 2)
    r1 = pivot + (range_1d * 1.1 / 12)
    s3 = pivot - (range_1d * 1.1 / 4)
    s4 = pivot - (range_1d * 1.1 / 2)
    s1 = pivot - (range_1d * 1.1 / 12)
    
    # Align daily levels to 4h timeframe (wait for daily close)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Choppiness Index (CHOP) - 14 period
    def calculate_chop(high, low, close, period=14):
        atr = np.zeros(len(close))
        tr = np.zeros(len(close))
        for i in range(1, len(close)):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i-1])
            lc = abs(low[i] - close[i-1])
            tr[i] = max(hl, hc, lc)
        # Smooth TR with Wilder's smoothing (alpha = 1/period)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        # Sum of TR over period
        sum_tr = np.zeros(len(close))
        for i in range(period, len(close)):
            if i == period:
                sum_tr[i] = np.sum(tr[1:i+1])
            else:
                sum_tr[i] = sum_tr[i-1] + tr[i] - tr[i-period+1]
        # Max high - min low over period
        max_high = np.zeros(len(close))
        min_low = np.zeros(len(close))
        for i in range(period, len(close)):
            max_high[i] = np.max(high[i-period+1:i+1])
            min_low[i] = np.min(low[i-period+1:i+1])
        # CHOP formula
        chop = np.full(len(close), 50.0)
        for i in range(period, len(close)):
            if sum_tr[i] > 0 and (max_high[i] - min_low[i]) > 0:
                chop[i] = 100 * np.log10(sum_tr[i] / (max_high[i] - min_low[i])) / np.log10(period)
        return chop
    
    chop = calculate_chop(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(volume[i]) or
            np.isnan(chop[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(r1_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(s1_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Chop regime: 38.2 < CHOP < 61.8 (ranging market)
        chop_in_range = (chop[i] > 38.2) and (chop[i] < 61.8)
        
        if position == 1:  # Long position
            # Exit: price touches S1 or breaks below S4 (failed mean reversion)
            if close[i] <= s1_aligned[i] or close[i] >= s4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price touches R1 or breaks above R4 (failed mean reversion)
            if close[i] >= r1_aligned[i] or close[i] <= r4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for mean reversion at pivot levels with volume and chop filter
            long_setup = (close[i] <= s3_aligned[i]) and volume_confirmed and chop_in_range
            short_setup = (close[i] >= r3_aligned[i]) and volume_confirmed and chop_in_range
            
            if long_setup:
                position = 1
                signals[i] = 0.25
            elif short_setup:
                position = -1
                signals[i] = -0.25
    
    return signals