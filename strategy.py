#!/usr/bin/env python3
# 6h_1d_weekly_pivot_reversal_v1
# Hypothesis: 6-hour reversal at weekly pivot levels with daily volume confirmation.
# Long when price bounces off weekly S1/S2/S3 with volume surge.
# Short when price reverses from weekly R1/R2/R3 with volume surge.
# Uses weekly pivot points calculated from prior week's OHLC.
# Designed to capture mean reversions at key weekly levels in both bull and bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_weekly_pivot_reversal_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly pivot points from prior week
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot: (H + L + C) / 3
    # Support/resistance levels:
    # R2 = P + (H - L)
    # R1 = 2*P - L
    # P = (H + L + C) / 3
    # S1 = 2*P - H
    # S2 = P - (H - L)
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    pivot_w = (high_w + low_w + close_w) / 3
    r2_w = pivot_w + (high_w - low_w)
    r1_w = 2 * pivot_w - low_w
    s1_w = 2 * pivot_w - high_w
    s2_w = pivot_w - (high_w - low_w)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_w_aligned = align_htf_to_ltf(prices, df_1w, pivot_w)
    r1_w_aligned = align_htf_to_ltf(prices, df_1w, r1_w)
    r2_w_aligned = align_htf_to_ltf(prices, df_1w, r2_w)
    s1_w_aligned = align_htf_to_ltf(prices, df_1w, s1_w)
    s2_w_aligned = align_htf_to_ltf(prices, df_1w, s2_w)
    
    # Daily volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_1d = df_1d['volume'].values
    vol_avg_20 = np.full(len(vol_1d), np.nan)
    for i in range(20, len(vol_1d)):
        vol_avg_20[i] = np.mean(vol_1d[i-20:i])
    
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        price = close[i]
        vol = volume[i]
        vol_avg = vol_avg_20_aligned[i]
        pivot = pivot_w_aligned[i]
        r1 = r1_w_aligned[i]
        r2 = r2_w_aligned[i]
        s1 = s1_w_aligned[i]
        s2 = s2_w_aligned[i]
        
        if np.isnan(vol_avg) or np.isnan(pivot) or np.isnan(r1) or np.isnan(r2) or np.isnan(s1) or np.isnan(s2):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        vol_surge = vol > 1.5 * vol_avg
        
        if position == 1:  # Long position
            # Exit if price reaches pivot or shows weakness at resistance
            if price >= pivot or (price >= r1 and vol_surge):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if price reaches pivot or shows strength at support
            if price <= pivot or (price <= s1 and vol_surge):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for bounce off support with volume
            if price <= s2 and vol_surge and price > s1:
                position = 1
                signals[i] = 0.25
            # Look for rejection at resistance with volume
            elif price >= r2 and vol_surge and price < r1:
                position = -1
                signals[i] = -0.25
    
    return signals