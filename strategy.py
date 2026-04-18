#!/usr/bin/env python3
"""
4h_Pivot_R1_S1_Breakout_Volume_Filter
Hypothesis: Use 12h and 1d Pivot R1/S1 levels for directional bias on 4h timeframe with volume confirmation (1.5x 20-period average) and ADX trend filter (ADX > 25 on 12h) to avoid chop. Enter long on breakout above both R1 levels, short on breakdown below both S1 levels. Exit on reversal or trend filter failure. Position size fixed at 0.25. Designed for ~25-35 trades/year to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend bias and ADX filter
    df_12h = get_htf_data(prices, '12h')
    
    # Get 1d data for entry levels
    df_1d = get_htf_data(prices, '1d')
    
    # 12h calculations for trend bias
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Previous 12h's OHLC for Pivot calculation
    prev_close_12h = np.roll(close_12h, 1)
    prev_high_12h = np.roll(high_12h, 1)
    prev_low_12h = np.roll(low_12h, 1)
    prev_close_12h[0] = close_12h[0]
    prev_high_12h[0] = high_12h[0]
    prev_low_12h[0] = low_12h[0]
    
    # 12h Pivot levels: P = (H+L+C)/3, R1 = 2*P - L, S1 = 2*P - H
    pivot_12h = (prev_high_12h + prev_low_12h + prev_close_12h) / 3.0
    r1_12h = 2 * pivot_12h - prev_low_12h
    s1_12h = 2 * pivot_12h - prev_high_12h
    
    # 1d calculations for entry
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Previous day's OHLC for Pivot calculation
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d[0] = close_1d[0]
    prev_high_1d[0] = high_1d[0]
    prev_low_1d[0] = low_1d[0]
    
    # 1d Pivot levels: P = (H+L+C)/3, R1 = 2*P - L, S1 = 2*P - H
    pivot_1d = (prev_high_1d + prev_low_1d + prev_close_1d) / 3.0
    r1_1d = 2 * pivot_1d - prev_low_1d
    s1_1d = 2 * pivot_1d - prev_high_1d
    
    # 12h ADX for trend strength filter (avoid chop)
    high_12h_arr = df_12h['high'].values
    low_12h_arr = df_12h['low'].values
    close_12h_arr = df_12h['close'].values
    
    # True Range
    tr1 = np.maximum(high_12h_arr - low_12h_arr, np.abs(high_12h_arr - np.roll(close_12h_arr, 1)))
    tr2 = np.abs(np.roll(close_12h_arr, 1) - low_12h_arr)
    tr = np.maximum(tr1, tr2)
    tr[0] = high_12h_arr[0] - low_12h_arr[0]
    
    # Directional Movement
    up_move = np.maximum(high_12h_arr - np.roll(high_12h_arr, 1), 0)
    down_move = np.maximum(np.roll(low_12h_arr, 1) - low_12h_arr, 0)
    up_move[0] = 0
    down_move[0] = 0
    
    # Smoothed values
    tr_period = 14
    tr_smooth = np.zeros_like(tr)
    tr_smooth[tr_period] = np.nansum(tr[1:tr_period+1]) if not np.isnan(tr).all() else 0
    for i in range(tr_period + 1, len(tr)):
        tr_smooth[i] = tr_smooth[i-1] - (tr_smooth[i-1] / tr_period) + tr[i]
    
    up_smooth = np.zeros_like(up_move)
    down_smooth = np.zeros_like(down_move)
    up_smooth[tr_period] = np.nansum(up_move[1:tr_period+1]) if not np.isnan(up_move).all() else 0
    down_smooth[tr_period] = np.nansum(down_move[1:tr_period+1]) if not np.isnan(down_move).all() else 0
    for i in range(tr_period + 1, len(up_move)):
        up_smooth[i] = up_smooth[i-1] - (up_smooth[i-1] / tr_period) + up_move[i]
        down_smooth[i] = down_smooth[i-1] - (down_smooth[i-1] / tr_period) + down_move[i]
    
    # Directional Indicators
    plus_di = 100 * up_smooth / tr_smooth
    minus_di = 100 * down_smooth / tr_smooth
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    
    # ADX
    adx_period = 14
    adx = np.zeros_like(dx)
    adx[2*adx_period] = np.nanmean(dx[adx_period:2*adx_period+1]) if not np.isnan(dx).all() else 0
    for i in range(2*adx_period + 1, len(dx)):
        adx[i] = (adx[i-1] * (adx_period - 1) + dx[i]) / adx_period
    
    # Align all higher timeframe data to 4h
    r1_12h_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for ADX and averages
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_12h_aligned[i]) or np.isnan(s1_12h_aligned[i]) or 
            np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(adx_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > 1.5 * vol_ma[i] if not np.isnan(vol_ma[i]) else False
        
        # Trend filter: 12h ADX > 25 to avoid chop
        trend_filter = adx_12h_aligned[i] > 25 if not np.isnan(adx_12h_aligned[i]) else False
        
        # Only trade during active session
        in_session = session_mask[i]
        
        if position == 0:
            # Long: price breaks above 12h R1 and 1d R1 with volume and trend filter during session
            if close[i] > r1_12h_aligned[i] and close[i] > r1_1d_aligned[i] and vol_confirm and trend_filter and in_session:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12h S1 and 1d S1 with volume and trend filter during session
            elif close[i] < s1_12h_aligned[i] and close[i] < s1_1d_aligned[i] and vol_confirm and trend_filter and in_session:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns below 12h R1 or 1d R1 or trend filter fails or outside session
            if close[i] < r1_12h_aligned[i] or close[i] < r1_1d_aligned[i] or not trend_filter or not in_session:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above 12h S1 or 1d S1 or trend filter fails or outside session
            if close[i] > s1_12h_aligned[i] or close[i] > s1_1d_aligned[i] or not trend_filter or not in_session:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Pivot_R1_S1_Breakout_Volume_Filter"
timeframe = "4h"
leverage = 1.0