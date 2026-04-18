#!/usr/bin/env python3
"""
12h_1D_1W_Camarilla_R1S1_Breakout_Volume_Trend_v3
Hypothesis: Use daily and weekly Camarilla R1/S1 levels for directional bias on 12h timeframe, with volume confirmation (>1.5x 20-period average) and trend filter from weekly ADX > 25. Enter only on breakouts of both daily and weekly levels. Exit on reversal of either level. Position size fixed at 0.25 to balance risk and reward. Designed for 15-35 trades/year to avoid fee drag, targeting 60-140 total trades over 4 years.
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
    
    # Get 1d data for daily Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    
    # Get 1w data for weekly trend filter and levels
    df_1w = get_htf_data(prices, '1w')
    
    # 1d calculations for entry levels (using previous day's OHLC)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d[0] = close_1d[0]
    prev_high_1d[0] = high_1d[0]
    prev_low_1d[0] = low_1d[0]
    
    # 1d Camarilla levels: R1 = close + (high-low)*1.1/12, S1 = close - (high-low)*1.1/12
    range_1d = prev_high_1d - prev_low_1d
    r1_1d = prev_close_1d + range_1d * 1.1 / 12
    s1_1d = prev_close_1d - range_1d * 1.1 / 12
    
    # 1w calculations for trend filter and levels
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Previous week's OHLC for Camarilla calculation
    prev_close_1w = np.roll(close_1w, 1)
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w[0] = close_1w[0]
    prev_high_1w[0] = high_1w[0]
    prev_low_1w[0] = low_1w[0]
    
    # 1w Camarilla levels: R1 = close + (high-low)*1.1/12, S1 = close - (high-low)*1.1/12
    range_1w = prev_high_1w - prev_low_1w
    r1_1w = prev_close_1w + range_1w * 1.1 / 12
    s1_1w = prev_close_1w - range_1w * 1.1 / 12
    
    # 1w ADX for trend strength filter (avoid chop)
    high_1w_arr = df_1w['high'].values
    low_1w_arr = df_1w['low'].values
    close_1w_arr = df_1w['close'].values
    
    # True Range
    tr1 = np.maximum(high_1w_arr - low_1w_arr, np.abs(high_1w_arr - np.roll(close_1w_arr, 1)))
    tr2 = np.abs(np.roll(close_1w_arr, 1) - low_1w_arr)
    tr = np.maximum(tr1, tr2)
    tr[0] = high_1w_arr[0] - low_1w_arr[0]
    
    # Directional Movement
    up_move = np.maximum(high_1w_arr - np.roll(high_1w_arr, 1), 0)
    down_move = np.maximum(np.roll(low_1w_arr, 1) - low_1w_arr, 0)
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
    
    # Align all higher timeframe data to 12h
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Precompute volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for ADX and averages
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) or 
            np.isnan(adx_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above both 1d R1 and 1w R1 with volume and trend filter
            if (close[i] > r1_1d_aligned[i] and close[i] > r1_1w_aligned[i] and 
                vol_confirm[i] and adx_1w_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below both 1d S1 and 1w S1 with volume and trend filter
            elif (close[i] < s1_1d_aligned[i] and close[i] < s1_1w_aligned[i] and 
                  vol_confirm[i] and adx_1w_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns below either 1d R1 or 1w R1
            if close[i] < r1_1d_aligned[i] or close[i] < r1_1w_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above either 1d S1 or 1w S1
            if close[i] > s1_1d_aligned[i] or close[i] > s1_1w_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1D_1W_Camarilla_R1S1_Breakout_Volume_Trend_v3"
timeframe = "12h"
leverage = 1.0