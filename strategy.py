#!/usr/bin/env python3
"""
1d_1W_Camarilla_R1S1_Breakout_Volume_Sparse_v2
Hypothesis: Daily close breaking weekly Camarilla R1/S1 with volume > 2x 20-day average and weekly ADX > 25.
Targets only strong breakouts in trending conditions to keep trades < 10/year. Works in bull/bear via ADX filter.
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
    
    # Get weekly data for context and levels
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly calculations
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Previous week's OHLC for Camarilla calculation
    prev_close = np.roll(close_1w, 1)
    prev_high = np.roll(high_1w, 1)
    prev_low = np.roll(low_1w, 1)
    prev_close[0] = close_1w[0]
    prev_high[0] = high_1w[0]
    prev_low[0] = low_1w[0]
    
    # Weekly Camarilla levels: R1 = close + (high-low)*1.1/12, S1 = close - (high-low)*1.1/12
    range_1w = prev_high - prev_low
    r1_1w = prev_close + range_1w * 1.1 / 12
    s1_1w = prev_close - range_1w * 1.1 / 12
    
    # Weekly ADX for trend strength (avoid chop)
    tr1 = np.maximum(high_1w - low_1w, np.abs(high_1w - np.roll(close_1w, 1)))
    tr2 = np.abs(np.roll(close_1w, 1) - low_1w)
    tr = np.maximum(tr1, tr2)
    tr[0] = high_1w[0] - low_1w[0]
    
    up_move = np.maximum(high_1w - np.roll(high_1w, 1), 0)
    down_move = np.maximum(np.roll(low_1w, 1) - low_1w, 0)
    up_move[0] = 0
    down_move[0] = 0
    
    tr_period = 10
    tr_smooth = np.zeros_like(tr)
    if len(tr) > tr_period:
        tr_smooth[tr_period] = np.nansum(tr[1:tr_period+1]) if not np.isnan(tr).all() else 0
        for i in range(tr_period + 1, len(tr)):
            tr_smooth[i] = tr_smooth[i-1] - (tr_smooth[i-1] / tr_period) + tr[i]
    
    up_smooth = np.zeros_like(up_move)
    down_smooth = np.zeros_like(down_move)
    if len(up_move) > tr_period:
        up_smooth[tr_period] = np.nansum(up_move[1:tr_period+1]) if not np.isnan(up_move).all() else 0
        down_smooth[tr_period] = np.nansum(down_move[1:tr_period+1]) if not np.isnan(down_move).all() else 0
        for i in range(tr_period + 1, len(up_move)):
            up_smooth[i] = up_smooth[i-1] - (up_smooth[i-1] / tr_period) + up_move[i]
            down_smooth[i] = down_smooth[i-1] - (down_smooth[i-1] / tr_period) + down_move[i]
    
    plus_di = 100 * up_smooth / tr_smooth
    minus_di = 100 * down_smooth / tr_smooth
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    
    adx_period = 10
    adx = np.zeros_like(dx)
    if len(dx) > 2 * adx_period:
        adx[2*adx_period] = np.nanmean(dx[adx_period:2*adx_period+1]) if not np.isnan(dx).all() else 0
        for i in range(2*adx_period + 1, len(dx)):
            adx[i] = (adx[i-1] * (adx_period - 1) + dx[i]) / adx_period
    
    # Align weekly data to daily
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Volume confirmation: current volume > 2x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # need enough for weekly ADX and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) or 
            np.isnan(adx_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Trend filter: ADX > 25 to avoid chop
        trend_filter = adx_1w_aligned[i] > 25
        
        if position == 0:
            # Long: price closes above weekly R1 with volume and trend filter
            if close[i] > r1_1w_aligned[i] and vol_confirm and trend_filter:
                signals[i] = 0.25
                position = 1
            # Short: price closes below weekly S1 with volume and trend filter
            elif close[i] < s1_1w_aligned[i] and vol_confirm and trend_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns below weekly R1 or trend fails
            if close[i] < r1_1w_aligned[i] or not trend_filter:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above weekly S1 or trend fails
            if close[i] > s1_1w_aligned[i] or not trend_filter:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1W_Camarilla_R1S1_Breakout_Volume_Sparse_v2"
timeframe = "1d"
leverage = 1.0