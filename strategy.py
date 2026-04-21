#!/usr/bin/env python3
"""
4h_PivotPoint_R1S1_Breakout_VolumeTrend
Hypothesis: Daily pivot points (R1/S1) act as key institutional levels. Breakouts with volume confirmation and 1-week EMA34 trend filter capture momentum while reducing whipsaw. Works in bull/bear markets by filtering breakouts with higher timeframe trend. Target 20-40 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Load daily data once for pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate daily pivot points (R1, S1)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r1 = pivot + (range_1d * 1.1 / 12)  # R1 = PP + (H-L)*1.1/12
    s1 = pivot - (range_1d * 1.1 / 12)  # S1 = PP - (H-L)*1.1/12
    
    # Align daily pivot levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Load weekly data once for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema34_1w = np.zeros_like(close_1w)
    ema34_1w[0] = close_1w[0]
    alpha = 2.0 / (34 + 1)
    for i in range(1, len(close_1w)):
        ema34_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema34_1w[i-1]
    
    # Align weekly EMA34 to 4h timeframe
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume filter: volume > 1.5x 20-period average
    volume = prices['volume'].values
    volume_avg = np.full(n, np.nan)
    for i in range(n):
        if i < 19:
            volume_avg[i] = np.mean(volume[0:i+1]) if i >= 0 else volume[i]
        else:
            volume_avg[i] = np.mean(volume[i-19:i+1])
    volume_filter = volume > (1.5 * volume_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if NaN in critical values
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(ema34_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema34 = ema34_1w_aligned[i]
        vol_confirm = volume_filter[i]
        
        # Calculate ATR for stoploss (20-period)
        if i >= 20:
            tr_values = []
            for j in range(1, 21):
                idx = i - j
                if idx >= 0:
                    tr = max(prices['high'].iloc[idx] - prices['low'].iloc[idx], 
                             abs(prices['high'].iloc[idx] - prices['close'].iloc[idx-1]), 
                             abs(prices['low'].iloc[idx] - prices['close'].iloc[idx-1]))
                    tr_values.append(tr)
            atr = np.mean(tr_values) if tr_values else 0
        else:
            atr = 0
        
        # Stoploss: 2.5 * ATR from entry
        if position == 1 and price < entry_price - 2.5 * atr:
            signals[i] = 0.0
            position = 0
            continue
        elif position == -1 and price > entry_price + 2.5 * atr:
            signals[i] = 0.0
            position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with volume confirmation in uptrend (price > weekly EMA34)
            if price > r1_val and vol_confirm and price > ema34:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below S1 with volume confirmation in downtrend (price < weekly EMA34)
            elif price < s1_val and vol_confirm and price < ema34:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price returns to S1 or trend breaks
            if price < s1_val or price < ema34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to R1 or trend breaks
            if price > r1_val or price > ema34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_PivotPoint_R1S1_Breakout_VolumeTrend"
timeframe = "4h"
leverage = 1.0