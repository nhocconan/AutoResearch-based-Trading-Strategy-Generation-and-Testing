#!/usr/bin/env python3
# 12h_camarilla_pivot_breakout_volume_v1
# Hypothesis: 12h strategy using daily Camarilla pivot levels (from 1d HTF) for breakout entries, volume confirmation (>1.5x 20-bar avg volume), and trend alignment via 1d EMA(50). Enters long when price breaks above daily R4 with volume and price > 1d EMA(50); enters short when price breaks below daily S4 with volume and price < 1d EMA(50). Exits on opposite pivot level touch (R3/S3) or close beyond R5/S5. Uses discrete sizing (0.25) to limit fee churn. Target: 12-37 trades/year (50-150 total over 4 years). Daily pivots provide structural support/resistance that works in bull/bear markets; volume confirms breakout conviction; 1d EMA filters counter-trend noise.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_breakout_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period = 10 days of 12h bars)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Multi-timeframe: 1d EMA(50) trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    close_1d_s = pd.Series(close_1d)
    ema_50_1d = close_1d_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Multi-timeframe: daily Camarilla pivot levels (from 1d HTF)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d_vals) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r3 = pivot + (range_1d * 1.1 / 4)
    r4 = pivot + (range_1d * 1.1 / 2)
    r5 = pivot + (range_1d * 1.1)
    s3 = pivot - (range_1d * 1.1 / 4)
    s4 = pivot - (range_1d * 1.1 / 2)
    s5 = pivot - (range_1d * 1.1)
    
    # Align daily levels to 12h timeframe (wait for daily close)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r5_aligned = align_htf_to_ltf(prices, df_1d, r5)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    s5_aligned = align_htf_to_ltf(prices, df_1d, s5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(volume[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(r5_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(s5_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Trend filters
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price touches daily R3 or breaks above R5 (failed breakout)
            if close[i] <= r3_aligned[i] or close[i] >= r5_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price touches daily S3 or breaks below S5 (failed breakout)
            if close[i] >= s3_aligned[i] or close[i] <= s5_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for daily breakout with volume and trend alignment
            bullish_breakout = (close[i] > r4_aligned[i]) and volume_confirmed and uptrend
            bearish_breakout = (close[i] < s4_aligned[i]) and volume_confirmed and downtrend
            
            if bullish_breakout:
                position = 1
                signals[i] = 0.25
            elif bearish_breakout:
                position = -1
                signals[i] = -0.25
    
    return signals