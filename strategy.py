#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot points
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points from previous week
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    
    # For each day, calculate pivot based on previous day's OHLC
    pivot = np.full_like(close_1d, np.nan)
    r1 = np.full_like(close_1d, np.nan)
    s1 = np.full_like(close_1d, np.nan)
    r2 = np.full_like(close_1d, np.nan)
    s2 = np.full_like(close_1d, np.nan)
    r3 = np.full_like(close_1d, np.nan)
    s3 = np.full_like(close_1d, np.nan)
    
    for i in range(1, len(close_1d)):
        ph = high_1d[i-1]
        pl = low_1d[i-1]
        pc = close_1d[i-1]
        
        p = (ph + pl + pc) / 3.0
        pivot[i] = p
        r1[i] = 2 * p - pl
        s1[i] = 2 * p - ph
        r2[i] = p + (ph - pl)
        s2[i] = p - (ph - pl)
        r3[i] = ph + 2 * (p - pl)
        s3[i] = pl - 2 * (ph - p)
    
    # Get weekly data for trend filter (EMA 34)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    if len(close_1w) >= 34:
        ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False).mean().values
    else:
        ema_1w = np.full_like(close_1w, np.nan)
    
    # Align all data to 12h timeframe
    pivot_12h = align_htf_to_ltf(prices, df_1d, pivot)
    r1_12h = align_htf_to_ltf(prices, df_1d, r1)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1)
    r2_12h = align_htf_to_ltf(prices, df_1d, r2)
    s2_12h = align_htf_to_ltf(prices, df_1d, s2)
    r3_12h = align_htf_to_ltf(prices, df_1d, r3)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3)
    ema_1w_12h = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation: volume > 1.5x 24-period average (2 days of 12h data)
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 24
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(1, vol_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_12h[i]) or np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or
            np.isnan(r2_12h[i]) or np.isnan(s2_12h[i]) or np.isnan(r3_12h[i]) or
            np.isnan(s3_12h[i]) or np.isnan(ema_1w_12h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: price above/below weekly EMA34
        uptrend = close[i] > ema_1w_12h[i]
        downtrend = close[i] < ema_1w_12h[i]
        
        if position == 0:
            # Long: price touches S1 support in uptrend with volume
            if (abs(close[i] - s1_12h[i]) < 0.001 * close[i] or close[i] >= s1_12h[i]) and \
               uptrend and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price touches R1 resistance in downtrend with volume
            elif (abs(close[i] - r1_12h[i]) < 0.001 * close[i] or close[i] <= r1_12h[i]) and \
                 downtrend and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price reaches R1 or trend changes
            if close[i] >= r1_12h[i] or not uptrend:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches S1 or trend changes
            if close[i] <= s1_12h[i] or not downtrend:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Pivot_S1_R1_Bounce_WeeklyTrend"
timeframe = "12h"
leverage = 1.0