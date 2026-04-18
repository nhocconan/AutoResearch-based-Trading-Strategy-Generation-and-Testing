#!/usr/bin/env python3
"""
4h_Wick_Reversal_V1
Hypothesis: Identify rejection at daily support/resistance via long upper/lower wicks. 
Long when price closes in lower 25% of daily range with long upper wick (bearish rejection fails). 
Short when price closes in upper 25% of daily range with long lower wick (bullish rejection fails). 
Requires volume > 1.5x 20-period average and 4h ADX > 20 to avoid chop. 
Targets 20-40 trades/year with discrete sizing (0.25) to minimize fee drag. 
Works in bull/bear via rejection logic that captures failed breakouts in any regime.
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
    
    # Get daily data for support/resistance and wick analysis
    df_1d = get_htf_data(prices, '1d')
    
    # Get 4h data for ADX filter
    df_4h = get_htf_data(prices, '4h')
    
    # Daily calculations
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily range and position
    daily_range = high_1d - low_1d
    close_position = (close_1d - low_1d) / daily_range  # 0 = low, 1 = high
    
    # Wick calculations: upper wick = high - close, lower wick = close - low
    upper_wick = high_1d - close_1d
    lower_wick = close_1d - low_1d
    
    # Long wick conditions: wick > 60% of daily range
    long_upper_wick = upper_wick > 0.6 * daily_range
    long_lower_wick = lower_wick > 0.6 * daily_range
    
    # Reversal signals: 
    # Long setup: price rejected from high (long upper wick) but closed in lower 25% of range
    long_setup = long_upper_wick & (close_position < 0.25)
    # Short setup: price rejected from low (long lower wick) but closed in upper 25% of range
    short_setup = long_lower_wick & (close_position > 0.75)
    
    # 4h ADX for trend strength filter (avoid chop)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
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
    
    # Smoothed values with proper smoothing
    tr_period = 14
    tr_smooth = np.zeros_like(tr)
    tr_smooth[tr_period] = np.nansum(tr[1:tr_period+1]) if tr_period < len(tr) else 0
    for i in range(tr_period + 1, len(tr)):
        tr_smooth[i] = tr_smooth[i-1] - (tr_smooth[i-1] / tr_period) + tr[i]
    
    up_smooth = np.zeros_like(up_move)
    down_smooth = np.zeros_like(down_move)
    up_smooth[tr_period] = np.nansum(up_move[1:tr_period+1]) if tr_period < len(up_move) else 0
    down_smooth[tr_period] = np.nansum(down_move[1:tr_period+1]) if tr_period < len(down_move) else 0
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
    if 2*adx_period < len(dx):
        adx[2*adx_period] = np.nanmean(dx[adx_period:2*adx_period+1]) if not np.isnan(dx[adx_period:2*adx_period+1]).all() else 0
        for i in range(2*adx_period + 1, len(dx)):
            adx[i] = (adx[i-1] * (adx_period - 1) + dx[i]) / adx_period
    
    # Align all higher timeframe data to 4h
    long_setup_aligned = align_htf_to_ltf(prices, df_1d, long_setup.astype(float))
    short_setup_aligned = align_htf_to_ltf(prices, df_1d, short_setup.astype(float))
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx)
    volume_ma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # need enough for ADX and averages
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(long_setup_aligned[i]) or np.isnan(short_setup_aligned[i]) or 
            np.isnan(adx_4h_aligned[i]) or np.isnan(volume_ma_4h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * volume_ma_4h[i]
        
        # Trend filter: ADX > 20 to avoid chop
        trend_filter = adx_4h_aligned[i] > 20
        
        if position == 0:
            # Long: long upper wick rejection + close in lower 25% + volume + trend
            if long_setup_aligned[i] and vol_confirm and trend_filter:
                signals[i] = 0.25
                position = 1
            # Short: long lower wick rejection + close in upper 25% + volume + trend
            elif short_setup_aligned[i] and vol_confirm and trend_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: reverse signal appears or trend fails
            if short_setup_aligned[i] or not trend_filter:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: reverse signal appears or trend fails
            if long_setup_aligned[i] or not trend_filter:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Wick_Reversal_V1"
timeframe = "4h"
leverage = 1.0