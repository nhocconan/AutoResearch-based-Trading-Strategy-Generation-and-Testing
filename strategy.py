#!/usr/bin/env python3
# 12h_1d_camarilla_pivot_volume_v1
# Strategy: 12h Camarilla pivot level touch with 1d volume confirmation and 1w trend filter
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels act as strong support/resistance. Long at S1/S2 with 1d volume spike and 1w uptrend; short at R1/R2 with volume spike and 1w downtrend. Works in both bull/bear by trading mean reversion at institutional levels with trend filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_pivot_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    prev_close = df_1d['close'].values
    
    # Calculate Camarilla levels for previous day
    # Pivot = (H + L + C) / 3
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Resistance levels
    r1 = pivot + (range_hl * 1.0833)
    r2 = pivot + (range_hl * 1.1666)
    r3 = pivot + (range_hl * 1.2500)
    r4 = pivot + (range_hl * 1.3333)
    
    # Support levels
    s1 = pivot - (range_hl * 1.0833)
    s2 = pivot - (range_hl * 1.1666)
    s3 = pivot - (range_hl * 1.2500)
    s4 = pivot - (range_hl * 1.3333)
    
    # Align Camarilla levels to 12h
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # 1d volume spike detection (volume > 1.5x 20-period average)
    vol_ma = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_spike = df_1d['volume'].values > (vol_ma * 1.5)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    # 1w EMA50 trend filter
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(r1_aligned[i]) or np.isnan(r2_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(ema50_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: price relative to 1w EMA50
        uptrend = close[i] > ema50_1w_aligned[i]
        downtrend = close[i] < ema50_1w_aligned[i]
        
        # Entry conditions
        # Long: Price at S1/S2 level with volume spike and uptrend
        if vol_spike_aligned[i] and uptrend:
            if abs(close[i] - s1_aligned[i]) < (s1_aligned[i] * 0.002) or \
               abs(close[i] - s2_aligned[i]) < (s2_aligned[i] * 0.002):
                if position != 1:
                    position = 1
                    signals[i] = 0.25
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.25 if position == 1 else 0.0
        # Short: Price at R1/R2 level with volume spike and downtrend
        elif vol_spike_aligned[i] and downtrend:
            if abs(close[i] - r1_aligned[i]) < (r1_aligned[i] * 0.002) or \
               abs(close[i] - r2_aligned[i]) < (r2_aligned[i] * 0.002):
                if position != -1:
                    position = -1
                    signals[i] = -0.25
                else:
                    signals[i] = -0.25
            else:
                signals[i] = -0.25 if position == -1 else 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals