#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot bounce with 1d volume confirmation and trend filter.
# Camarilla levels provide precise support/resistance for mean reversion.
# Volume confirms institutional interest at pivot levels.
# 1d trend filter ensures we trade with higher timeframe momentum.
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for multi-timeframe analysis
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 12h Camarilla pivot levels (based on previous day)
    camarilla_h4 = np.full(n, np.nan)  # Resistance 1
    camarilla_l4 = np.full(n, np.nan)  # Support 1
    camarilla_h3 = np.full(n, np.nan)  # Resistance 2
    camarilla_l3 = np.full(n, np.nan)  # Support 2
    
    # Calculate pivots using previous day's OHLC
    for i in range(1, n):
        # Get previous day's OHLC (assuming 12h bars, 2 per day)
        prev_idx = i - 2
        if prev_idx >= 0:
            # We need daily OHLC, so we'll use the 1d data
            pass
    
    # Calculate Camarilla levels from 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_h4_1d = np.full(len(close_1d), np.nan)
    camarilla_l4_1d = np.full(len(close_1d), np.nan)
    camarilla_h3_1d = np.full(len(close_1d), np.nan)
    camarilla_l3_1d = np.full(len(close_1d), np.nan)
    camarilla_h2_1d = np.full(len(close_1d), np.nan)
    camarilla_l2_1d = np.full(len(close_1d), np.nan)
    camarilla_h1_1d = np.full(len(close_1d), np.nan)
    camarilla_l1_1d = np.full(len(close_1d), np.nan)
    pivot_1d = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):
        high_val = high_1d[i-1]  # previous day high
        low_val = low_1d[i-1]    # previous day low
        close_val = close_1d[i-1] # previous day close
        
        pivot = (high_val + low_val + close_val) / 3.0
        range_val = high_val - low_val
        
        camarilla_h1_1d[i] = close_val + range_val * 1.1 / 12
        camarilla_l1_1d[i] = close_val - range_val * 1.1 / 12
        camarilla_h2_1d[i] = close_val + range_val * 1.1 / 6
        camarilla_l2_1d[i] = close_val - range_val * 1.1 / 6
        camarilla_h3_1d[i] = close_val + range_val * 1.1 / 4
        camarilla_l3_1d[i] = close_val - range_val * 1.1 / 4
        camarilla_h4_1d[i] = close_val + range_val * 1.1 / 2
        camarilla_l4_1d[i] = close_val - range_val * 1.1 / 2
        pivot_1d[i] = pivot
    
    # Align Camarilla levels to 12h timeframe
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4_1d)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3_1d)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    
    # Calculate 1d average volume for confirmation
    vol_1d = df_1d['volume'].values
    avg_vol_1d = np.zeros(len(vol_1d))
    for i in range(20, len(vol_1d)):
        avg_vol_1d[i] = np.mean(vol_1d[i-20:i])
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    
    # Calculate 1d EMA for trend filter
    ema_1d = np.zeros(len(close_1d))
    for i in range(50, len(close_1d)):
        if i == 50:
            ema_1d[i] = np.mean(close_1d[:i])
        else:
            ema_1d[i] = close_1d[i] * 0.04 + ema_1d[i-1] * 0.96
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(h4_1d_aligned[i]) or np.isnan(l4_1d_aligned[i]) or 
            np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or
            np.isnan(pivot_aligned[i]) or np.isnan(avg_vol_1d_aligned[i]) or
            np.isnan(ema_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_vol_1d_aligned[i]
        h4 = h4_1d_aligned[i]
        l4 = l4_1d_aligned[i]
        h3 = h3_1d_aligned[i]
        l3 = l3_1d_aligned[i]
        pivot_val = pivot_aligned[i]
        ema_val = ema_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.3x average daily volume
        volume_confirm = vol > 1.3 * avg_vol
        
        if position == 0:
            # Long near support: price touches L3 or L4 with volume, above EMA (uptrend)
            if volume_confirm and price > ema_val:
                if abs(price - l3) < (h4 - l4) * 0.02 or abs(price - l4) < (h4 - l4) * 0.02:
                    position = 1
                    signals[i] = position_size
            # Short near resistance: price touches H3 or H4 with volume, below EMA (downtrend)
            elif volume_confirm and price < ema_val:
                if abs(price - h3) < (h4 - l4) * 0.02 or abs(price - h4) < (h4 - l4) * 0.02:
                    position = -1
                    signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches pivot or shows weakness
            if price >= pivot_val or vol < 0.7 * avg_vol:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price reaches pivot or shows weakness
            if price <= pivot_val or vol < 0.7 * avg_vol:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Camarilla_Pivot_Bounce_Volume_Trend"
timeframe = "12h"
leverage = 1.0