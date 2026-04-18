#!/usr/bin/env python3
"""
4h_Donchian_20_Breakout_Volume_Trend_Filter
Hypothesis: Use 20-period Donchian channel breakouts on 4h timeframe with volume confirmation (>1.5x 20-period average volume) and 4h ADX > 25 trend filter. Only trade during active session (08-20 UTC). Long on breakout above upper band, short on breakout below lower band. Exit on opposite band touch. Position size fixed at 0.25 to balance risk and reward. Designed for ~30-50 trades/year to avoid excessive fee drag while capturing strong trends in both bull and bear markets.
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
    
    # Get 4h data for Donchian channels and ADX
    df_4h = get_htf_data(prices, '4h')
    
    # 4h calculations
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Donchian Channel (20-period)
    period = 20
    upper = pd.Series(high_4h).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low_4h).rolling(window=period, min_periods=period).min().values
    
    # ADX calculation for trend strength
    # True Range
    tr1 = np.maximum(high_4h - low_4h, np.abs(high_4h - np.roll(close_4h, 1)))
    tr2 = np.abs(np.roll(close_4h, 1) - low_4h)
    tr = np.maximum(tr1, tr2)
    tr[0] = high_4h[0] - low_4h[0]
    
    # Directional Movement
    up_move = np.maximum(high_4h - np.roll(high_4h, 1), 0)
    down_move = np.maximum(np.roll(low_4h, 1) - low_4h, 0)
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
    
    # Align all higher timeframe data to 4h (actually 4h to 4h is identity, but keeping for consistency)
    upper_aligned = align_htf_to_ltf(prices, df_4h, upper)
    lower_aligned = align_htf_to_ltf(prices, df_4h, lower)
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx)
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 2*adx_period + 1)  # need enough for ADX and averages
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(adx_4h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: 4h ADX > 25 to avoid chop
        trend_filter = adx_4h_aligned[i] > 25
        
        # Only trade during active session
        in_session = session_mask[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian band with volume and trend filter
            if close[i] > upper_aligned[i] and vol_confirm and trend_filter and in_session:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian band with volume and trend filter
            elif close[i] < lower_aligned[i] and vol_confirm and trend_filter and in_session:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price touches or goes below lower Donchian band (mean reversion)
            # or trend filter fails or outside session
            if close[i] < lower_aligned[i] or not trend_filter or not in_session:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price touches or goes above upper Donchian band
            # or trend filter fails or outside session
            if close[i] > upper_aligned[i] or not trend_filter or not in_session:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_20_Breakout_Volume_Trend_Filter"
timeframe = "4h"
leverage = 1.0