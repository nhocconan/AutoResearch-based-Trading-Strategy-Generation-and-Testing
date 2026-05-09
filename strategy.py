#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WeeklyPivot_Consolidation_Breakout"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Previous week's OHLC for pivot calculation
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (standard formula)
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    r2_1w = pivot_1w + (high_1w - low_1w)
    s2_1w = pivot_1w - (high_1w - low_1w)
    r3_1w = high_1w + 2 * (pivot_1w - low_1w)
    s3_1w = low_1w - 2 * (high_1w - pivot_1w)
    
    # Shift by 1 week to avoid look-ahead
    pivot_1w_shift = np.roll(pivot_1w, 1)
    r1_1w_shift = np.roll(r1_1w, 1)
    s1_1w_shift = np.roll(s1_1w, 1)
    r2_1w_shift = np.roll(r2_1w, 1)
    s2_1w_shift = np.roll(s2_1w, 1)
    r3_1w_shift = np.roll(r3_1w, 1)
    s3_1w_shift = np.roll(s3_1w, 1)
    # Set first element to nan
    pivot_1w_shift[0] = np.nan
    r1_1w_shift[0] = np.nan
    s1_1w_shift[0] = np.nan
    r2_1w_shift[0] = np.nan
    s2_1w_shift[0] = np.nan
    r3_1w_shift[0] = np.nan
    s3_1w_shift[0] = np.nan
    
    # Align weekly pivot to 6h timeframe
    pivot_6h = align_htf_to_ltf(prices, df_1w, pivot_1w_shift)
    r1_6h = align_htf_to_ltf(prices, df_1w, r1_1w_shift)
    s1_6h = align_htf_to_ltf(prices, df_1w, s1_1w_shift)
    r2_6h = align_htf_to_ltf(prices, df_1w, r2_1w_shift)
    s2_6h = align_htf_to_ltf(prices, df_1w, s2_1w_shift)
    r3_6h = align_htf_to_ltf(prices, df_1w, r3_1w_shift)
    s3_6h = align_htf_to_ltf(prices, df_1w, s3_1w_shift)
    
    # Daily trend filter (EMA 34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_6h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: volume > 1.5x 50-period EMA
    vol_ema50 = pd.Series(volume).ewm(span=50, adjust=False, min_periods=50).mean().values
    vol_filter = volume > (1.5 * vol_ema50)
    
    # 6-period ATR for breakout confirmation
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(np.maximum(tr1, tr2), tr3)])
    atr6 = pd.Series(tr).ewm(span=6, adjust=False, min_periods=6).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(pivot_6h[i]) or np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or
            np.isnan(r2_6h[i]) or np.isnan(s2_6h[i]) or np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or
            np.isnan(ema_34_1d_6h[i]) or np.isnan(vol_ema50[i]) or np.isnan(atr6[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long breakout: price breaks above R2 with volume and above daily EMA
            if (price > r2_6h[i] and vol_filter[i] and price > ema_34_1d_6h[i] and 
                price > close[i-1] + 0.5 * atr6[i]):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below S2 with volume and below daily EMA
            elif (price < s2_6h[i] and vol_filter[i] and price < ema_34_1d_6h[i] and 
                  price < close[i-1] - 0.5 * atr6[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back to pivot or below
            if price <= pivot_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back to pivot or above
            if price >= pivot_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals