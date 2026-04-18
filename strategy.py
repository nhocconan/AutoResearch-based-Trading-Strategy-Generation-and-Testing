#!/usr/bin/env python3
"""
12h_1D_Camarilla_R1S1_Breakout_Volume_Tight_V3
Hypothesis: Use daily and weekly Camarilla R1/S1 for directional bias with 12h entry, requiring volume > 1.5x 20-period average and session filter (08-20 UTC). Add 1d ADX > 20 to avoid chop and ensure trending conditions. Tighten entry by requiring price to close beyond R1/S1 for confirmation, reducing false breakouts. Target 15-35 trades/year for low frequency and minimal fee drag. Works in bull/bear via volatility regime filter using 1d ADX.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for primary directional bias
    df_1d = get_htf_data(prices, '1d')
    
    # Get weekly data for additional context (optional filter)
    df_1w = get_htf_data(prices, '1w')
    
    # Daily calculations for bias
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = close_1d[0]
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    
    # Daily Camarilla levels: R1 = close + (high-low)*1.1/12, S1 = close - (high-low)*1.1/12
    range_1d = prev_high - prev_low
    r1_1d = prev_close + range_1d * 1.1 / 12
    s1_1d = prev_close - range_1d * 1.1 / 12
    
    # Weekly calculations for trend filter (ADX)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range for weekly
    tr1 = np.maximum(high_1w - low_1w, np.abs(high_1w - np.roll(close_1w, 1)))
    tr2 = np.abs(np.roll(close_1w, 1) - low_1w)
    tr = np.maximum(tr1, tr2)
    tr[0] = high_1w[0] - low_1w[0]
    
    # Directional Movement
    up_move = np.maximum(high_1w - np.roll(high_1w, 1), 0)
    down_move = np.maximum(np.roll(low_1w, 1) - low_1w, 0)
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
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # need enough for ADX and averages
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(adx_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > 1.5 * vol_ma[i] if not np.isnan(vol_ma[i]) else False
        
        # Trend filter: ADX > 20 to avoid chop
        trend_filter = adx_1w_aligned[i] > 20 if not np.isnan(adx_1w_aligned[i]) else False
        
        # Only trade during active session
        in_session = session_mask[i]
        
        if position == 0:
            # Long: price closes above daily R1 with volume and trend filter during session
            if close[i] > r1_1d_aligned[i] and vol_confirm and trend_filter and in_session:
                signals[i] = 0.25
                position = 1
            # Short: price closes below daily S1 with volume and trend filter during session
            elif close[i] < s1_1d_aligned[i] and vol_confirm and trend_filter and in_session:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns below daily R1 or trend filter fails or outside session
            if close[i] < r1_1d_aligned[i] or not trend_filter or not in_session:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above daily S1 or trend filter fails or outside session
            if close[i] > s1_1d_aligned[i] or not trend_filter or not in_session:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1D_Camarilla_R1S1_Breakout_Volume_Tight_V3"
timeframe = "12h"
leverage = 1.0