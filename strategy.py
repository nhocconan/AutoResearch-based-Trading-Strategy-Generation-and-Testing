#!/usr/bin/env python3
"""
6h_1D_1W_Camarilla_MultiTF_Structure_v1
Hypothesis: Use 1-day and 1-week Camarilla levels to define major support/resistance zones, with 6h entries on breakouts from these zones, filtered by volume spikes and ADX trend strength. This structure-based approach aims to capture major moves while avoiding chop, working in both bull and bear markets by following institutional levels. Target: 20-50 trades per year.
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
    
    # Get daily and weekly data for structural context
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Daily calculations for Camarilla levels (using previous day's OHLC)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Previous day's OHLC
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d[0] = close_1d[0]
    prev_high_1d[0] = high_1d[0]
    prev_low_1d[0] = low_1d[0]
    
    # Daily Camarilla levels: R3/S3 (stronger barriers)
    range_1d = prev_high_1d - prev_low_1d
    r3_1d = prev_close_1d + range_1d * 1.1 / 4
    s3_1d = prev_close_1d - range_1d * 1.1 / 4
    
    # Weekly calculations for Camarilla levels
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Previous week's OHLC
    prev_close_1w = np.roll(close_1w, 1)
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w[0] = close_1w[0]
    prev_high_1w[0] = high_1w[0]
    prev_low_1w[0] = low_1w[0]
    
    # Weekly Camarilla levels: R3/S3 (stronger barriers)
    range_1w = prev_high_1w - prev_low_1w
    r3_1w = prev_close_1w + range_1w * 1.1 / 4
    s3_1w = prev_close_1w - range_1w * 1.1 / 4
    
    # 6h ADX for trend strength filter
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # True Range
    tr1 = np.maximum(high_6h - low_6h, np.abs(high_6h - np.roll(close_6h, 1)))
    tr2 = np.abs(np.roll(close_6h, 1) - low_6h)
    tr = np.maximum(tr1, tr2)
    tr[0] = high_6h[0] - low_6h[0]
    
    # Directional Movement
    up_move = np.maximum(high_6h - np.roll(high_6h, 1), 0)
    down_move = np.maximum(np.roll(low_6h, 1) - low_6h, 0)
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
    
    # Align all higher timeframe data to 6h
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    adx_6h_aligned = align_htf_to_ltf(prices, df_6h, adx)
    
    # Precompute volume moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(r3_1w_aligned[i]) or np.isnan(s3_1w_aligned[i]) or 
            np.isnan(adx_6h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average
        vol_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Trend filter: ADX > 25 to ensure trending conditions
        trend_filter = adx_6h_aligned[i] > 25
        
        if position == 0:
            # Long: price breaks above both daily and weekly R3 with volume and trend
            if (close[i] > r3_1d_aligned[i] and close[i] > r3_1w_aligned[i] and 
                vol_confirm and trend_filter):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below both daily and weekly S3 with volume and trend
            elif (close[i] < s3_1d_aligned[i] and close[i] < s3_1w_aligned[i] and 
                  vol_confirm and trend_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns below daily R3 or weekly R3 or trend fails
            if (close[i] < r3_1d_aligned[i] or close[i] < r3_1w_aligned[i] or 
                not trend_filter):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above daily S3 or weekly S3 or trend fails
            if (close[i] > s3_1d_aligned[i] or close[i] > s3_1w_aligned[i] or 
                not trend_filter):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1D_1W_Camarilla_MultiTF_Structure_v1"
timeframe = "6h"
leverage = 1.0