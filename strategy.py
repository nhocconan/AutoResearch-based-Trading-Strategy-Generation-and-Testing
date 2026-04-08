#!/usr/bin/env python3
# 12h_1d_1w_camarilla_pivot_breakout_volume_v1
# Hypothesis: 12h Camarilla pivot (R4/S4) breakouts with volume confirmation (>2.0x 20-period average) and 1d/1w VWAP regime filter.
# Long: price > R4(1d) AND volume > 2.0x vol_MA20 AND VWAP_1d > VWAP_1w (bullish regime)
# Short: price < S4(1d) AND volume > 2.0x vol_MA20 AND VWAP_1d < VWAP_1w (bearish regime)
# Exit: price returns to VWAP_1d (mean reversion) OR breaks R3/S3 with volume confirmation
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 50-150 total trades over 4 years.
# Works in bull/bear via regime filter (1d vs 1w VWAP) and pivot structure.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_1w_camarilla_pivot_breakout_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume ratio: current vs 20-period SMA (min_periods=20)
    vol_sma = np.full(n, np.nan)
    for i in range(20, n):
        vol_sma[i] = np.mean(volume[i-20:i])
    vol_ratio = np.where(vol_sma > 0, volume / vol_sma, 0)
    
    # Get 1d data for Camarilla pivots and VWAP
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels from previous day
    camarilla_r4 = np.full(len(df_1d), np.nan)
    camarilla_r3 = np.full(len(df_1d), np.nan)
    camarilla_s3 = np.full(len(df_1d), np.nan)
    camarilla_s4 = np.full(len(df_1d), np.nan)
    vwap_1d = np.full(len(df_1d), np.nan)
    
    cum_vol = 0
    cum_pv = 0
    for i in range(len(df_1d)):
        typical_price = (high_1d[i] + low_1d[i] + close_1d[i]) / 3.0
        cum_pv += typical_price * volume[i]
        cum_vol += volume[i]
        vwap_1d[i] = cum_pv / cum_vol if cum_vol > 0 else typical_price
        
        if i > 0:
            prev_close = close_1d[i-1]
            prev_high = high_1d[i-1]
            prev_low = low_1d[i-1]
            range_val = prev_high - prev_low
            
            camarilla_r4[i] = prev_close + range_val * 1.1 / 2.0
            camarilla_r3[i] = prev_close + range_val * 1.1 / 4.0
            camarilla_s3[i] = prev_close - range_val * 1.1 / 4.0
            camarilla_s4[i] = prev_close - range_val * 1.1 / 2.0
    
    # Get 1w data for VWAP regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    vwap_1w = np.full(len(df_1w), np.nan)
    cum_vol_w = 0
    cum_pv_w = 0
    for i in range(len(df_1w)):
        typical_price = (high_1w[i] + low_1w[i] + close_1w[i]) / 3.0
        cum_pv_w += typical_price * volume_1w[i]
        cum_vol_w += volume_1w[i]
        vwap_1w[i] = cum_pv_w / cum_vol_w if cum_vol_w > 0 else typical_price
    
    # Align 1d indicators to 12h
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Align 1w VWAP to 12h
    vwap_1w_aligned = align_htf_to_ltf(prices, df_1w, vwap_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        vol_r = vol_ratio[i]
        price = close[i]
        
        if np.isnan(vol_r):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        r4 = camarilla_r4_aligned[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        s4 = camarilla_s4_aligned[i]
        vwap1d = vwap_1d_aligned[i]
        vwap1w = vwap_1w_aligned[i]
        
        if np.isnan(r4) or np.isnan(r3) or np.isnan(s3) or np.isnan(s4) or np.isnan(vwap1d) or np.isnan(vwap1w):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long
            # Exit: price returns to VWAP_1d OR breaks below R3 with volume
            if price <= vwap1d or (price < r3 and vol_r > 1.5):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price returns to VWAP_1d OR breaks above S3 with volume
            if price >= vwap1d or (price > s3 and vol_r > 1.5):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long: break above R4 with volume AND bullish regime (VWAP_1d > VWAP_1w)
            if price > r4 and vol_r > 2.0 and vwap1d > vwap1w:
                position = 1
                signals[i] = 0.25
            # Short: break below S4 with volume AND bearish regime (VWAP_1d < VWAP_1w)
            elif price < s4 and vol_r > 2.0 and vwap1d < vwap1w:
                position = -1
                signals[i] = -0.25
    
    return signals