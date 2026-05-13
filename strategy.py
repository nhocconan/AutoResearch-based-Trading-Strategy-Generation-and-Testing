#!/usr/bin/env python3
"""
6h_Pivot_Zone_Reversal_with_12hTrend_and_Volume
Hypothesis: Price reversals at 12-hour pivot zones (S1/R1, S2/R2) with 12h trend filter and volume confirmation.
In ranging markets, price tends to revert from S1/R1; in trending markets, breaks of S2/R2 with 12h trend continuation.
Works in both bull and bear via trend filter and pivot structure.
Target: 15-30 trades/year per symbol.
"""

name = "6h_Pivot_Zone_Reversal_with_12hTrend_and_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h pivot points (classic)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate pivot points from previous 12h bar
    prev_high = df_12h['high'].shift(1).values
    prev_low = df_12h['low'].shift(1).values
    prev_close = df_12h['close'].shift(1).values
    
    # Avoid look-ahead: use only prior bar data
    pivot = (prev_high + prev_low + prev_close) / 3.0
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    r2 = pivot + (prev_high - prev_low)
    s2 = pivot - (prev_high - prev_low)
    
    # Align to 6h timeframe (wait for 12h bar close)
    pivot_12h = align_htf_to_ltf(prices, df_12h, pivot)
    r1_12h = align_htf_to_ltf(prices, df_12h, r1)
    s1_12h = align_htf_to_ltf(prices, df_12h, s1)
    r2_12h = align_htf_to_ltf(prices, df_12h, r2)
    s2_12h = align_htf_to_ltf(prices, df_12h, s2)
    
    # 12h trend filter: EMA50 on 12h close
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_12h = df_12h['close'].values > ema_50_12h
    downtrend_12h = df_12h['close'].values < ema_50_12h
    uptrend_12h_aligned = align_htf_to_ltf(prices, df_12h, uptrend_12h)
    downtrend_12h_aligned = align_htf_to_ltf(prices, df_12h, downtrend_12h)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        px = close[i]
        vol_ok = volume_conf[i]
        uptrend = uptrend_12h_aligned[i]
        downtrend = downtrend_12h_aligned[i]
        
        if position == 0:
            # LONG setup: bounce from S1 with volume OR break above S2 with uptrend
            long_setup = False
            # Reversal from S1 (range-bound behavior)
            if px > s1_12h[i] and px < s1_12h[i] + (r1_12h[i] - s1_12h[i]) * 0.1:  # near S1
                if px > close[i-1] and vol_ok:  # bullish candle with volume
                    long_setup = True
            # Breakout above S2 with trend (trending behavior)
            elif px > s2_12h[i] and uptrend and vol_ok:
                long_setup = True
            
            # SHORT setup: rejection at R1 OR breakdown below R2 with downtrend
            short_setup = False
            # Rejection at R1 (range-bound behavior)
            if px < r1_12h[i] and px > r1_12h[i] - (r1_12h[i] - s1_12h[i]) * 0.1:  # near R1
                if px < close[i-1] and vol_ok:  # bearish candle with volume
                    short_setup = True
            # Breakdown below R2 with trend (trending behavior)
            elif px < r2_12h[i] and downtrend and vol_ok:
                short_setup = True
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
                
        elif position == 1:
            # EXIT LONG: rejection at R1 or stop below S1
            if px < r1_12h[i] or px < s1_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # EXIT SHORT: bounce at S1 or stop above R1
            if px > s1_12h[i] or px > r1_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals