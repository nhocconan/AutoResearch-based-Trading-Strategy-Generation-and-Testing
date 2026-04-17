#!/usr/bin/env python3
"""
Hypothesis: On 6h, price respects 1-week Camarilla pivot levels (H3, L3) as support/resistance.
We combine with volume confirmation and a daily EMA50 trend filter.
Long when price crosses above H3 with volume > 1.5x average and price above EMA50.
Short when price crosses below L3 with volume > 1.5x average and price below EMA50.
Exit when price returns to the 1-week midpoint (H4/L4) or on opposite signal.
Designed for 6h to work in trending and ranging markets with ~12-30 trades per year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Camarilla pivot levels
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Camarilla pivot levels from prior 1-week data
    # Using prior week's high, low, close to avoid look-ahead
    pweek_high = df_1w['high'].shift(1).values
    pweek_low = df_1w['low'].shift(1).values
    pweek_close = df_1w['close'].shift(1).values
    
    # Camarilla levels
    range_val = pweek_high - pweek_low
    h3 = pweek_close + range_val * 1.1 / 4
    l3 = pweek_close - range_val * 1.1 / 4
    h4 = pweek_close + range_val * 1.1 / 2
    l4 = pweek_close - range_val * 1.1 / 2
    
    # Calculate daily EMA50 for trend filter (use prior day's close to avoid look-ahead)
    df_1d = get_htf_data(prices, '1d')
    pday_close = df_1d['close'].shift(1).values
    ema_50 = pd.Series(pday_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all 1w levels to 6h timeframe (waits for 1w bar to close)
    h3_6h = align_htf_to_ltf(prices, df_1w, h3)
    l3_6h = align_htf_to_ltf(prices, df_1w, l3)
    h4_6h = align_htf_to_ltf(prices, df_1w, h4)
    l4_6h = align_htf_to_ltf(prices, df_1w, l4)
    
    # Align daily EMA50 to 6h
    ema_50_6h = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: 20-period volume MA on 6h
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 60  # warmup for EMA50 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(h3_6h[i]) or np.isnan(l3_6h[i]) or np.isnan(h4_6h[i]) or 
            np.isnan(l4_6h[i]) or np.isnan(ema_50_6h[i]) or np.isnan(volume_ma_20.iloc[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        
        if position == 0:
            # Long: price crosses above H3 with volume spike and above EMA50
            if price > h3_6h[i] and vol > 1.5 * vol_ma and price > ema_50_6h[i]:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below L3 with volume spike and below EMA50
            elif price < l3_6h[i] and vol > 1.5 * vol_ma and price < ema_50_6h[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to H4 (pivot resistance) or breaks below L3 (invalidates support)
            if price < h4_6h[i] or price < l3_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to L4 (pivot support) or breaks above H3 (invalidates resistance)
            if price > l4_6h[i] or price > h3_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1wCamarilla_H3L3_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0