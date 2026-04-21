#!/usr/bin/env python3
"""
1d_Pivot_R2S2_Breakout_Volume_ADXFilter
Hypothesis: Daily pivot levels R2/S2 act as key resistance/support zones. Breakouts above R2 or below S2 with volume and ADX trend confirmation capture strong moves. Works in bull markets via upside breakouts and in bear markets via downside breakdowns. Low trade frequency target (10-30/year) minimizes fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for ADX trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 14:
        return np.zeros(n)
    
    # Calculate weekly ADX (14)
    high_w = df_weekly['high'].values
    low_w = df_weekly['low'].values
    close_w = df_weekly['close'].values
    
    # True Range
    tr1 = np.abs(high_w[1:] - low_w[1:])
    tr2 = np.abs(high_w[1:] - close_w[:-1])
    tr3 = np.abs(low_w[1:] - close_w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # Directional Movement
    up_move = high_w[1:] - high_w[:-1]
    down_move = low_w[:-1] - low_w[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    # Smoothed values
    def smooth_wilder(arr, period):
        smoothed = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return smoothed
        smoothed[period-1] = np.nansum(arr[1:period])  # first seed
        for i in range(period, len(arr)):
            smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + arr[i]
        return smoothed
    
    tr_smooth = smooth_wilder(tr, 14)
    plus_dm_smooth = smooth_wilder(plus_dm, 14)
    minus_dm_smooth = smooth_wilder(minus_dm, 14)
    
    # DI and DX
    plus_di = np.where(tr_smooth != 0, (plus_dm_smooth / tr_smooth) * 100, 0)
    minus_di = np.where(tr_smooth != 0, (minus_dm_smooth / tr_smooth) * 100, 0)
    dx = np.where((plus_di + minus_di) != 0, np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
    
    # ADX: smoothed DX
    adx = np.full_like(dx, np.nan, dtype=float)
    if len(dx) >= 14:
        adx[13] = np.nanmean(dx[1:14])  # first seed
        for i in range(14, len(dx)):
            adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Align weekly ADX to daily
    adx_aligned = align_htf_to_ltf(prices, df_weekly, adx)
    
    # Load daily data for pivot points
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    high_d = df_daily['high'].values
    low_d = df_daily['low'].values
    close_d = df_daily['close'].values
    
    # Calculate daily pivot points (standard)
    # P = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    # R2 = P + (H - L)
    # S2 = P - (H - L)
    P = (high_d + low_d + close_d) / 3.0
    r2 = P + (high_d - low_d)
    s2 = P - (high_d - low_d)
    
    # Align daily pivot levels
    r2_aligned = align_htf_to_ltf(prices, df_daily, r2)
    s2_aligned = align_htf_to_ltf(prices, df_daily, s2)
    
    # Main timeframe data (1d)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.5x 20-day average
    volume_avg = np.full_like(volume, np.nan, dtype=float)
    for i in range(len(volume)):
        if i >= 20:
            volume_avg[i] = np.mean(volume[i-20:i])
        else:
            volume_avg[i] = np.mean(volume[:i+1]) if i > 0 else volume[i]
    volume_filter = volume > (1.5 * volume_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if NaN in critical values
        if (np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        r2 = r2_aligned[i]
        s2 = s2_aligned[i]
        adx_val = adx_aligned[i]
        vol_ok = volume_filter[i]
        
        # ADX filter: only trade when trending (ADX > 25)
        trend_ok = adx_val > 25
        
        if position == 0:
            # Long breakout above R2
            if price > r2 and vol_ok and trend_ok:
                signals[i] = 0.25
                position = 1
            # Short breakdown below S2
            elif price < s2 and vol_ok and trend_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to pivot point or breaks below S2 (failed breakout)
            P_daily = (high_d + low_d + close_d) / 3.0
            P_aligned = align_htf_to_ltf(prices, df_daily, P_daily)
            if not np.isnan(P_aligned[i]):
                if price < P_aligned[i] or price < s2:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to pivot point or breaks above R2 (failed breakdown)
            P_daily = (high_d + low_d + close_d) / 3.0
            P_aligned = align_htf_to_ltf(prices, df_daily, P_daily)
            if not np.isnan(P_aligned[i]):
                if price > P_aligned[i] or price > r2:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Pivot_R2S2_Breakout_Volume_ADXFilter"
timeframe = "1d"
leverage = 1.0