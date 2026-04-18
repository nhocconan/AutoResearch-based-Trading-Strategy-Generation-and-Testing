#!/usr/bin/env python3
"""
4h_Donchian_Breakout_With_Volume_and_Trend_Filter_v1
Hypothesis: Use 4h Donchian channel breakouts with volume confirmation and 1d ADX trend filter.
Long when price breaks above 20-period Donchian high with volume > 1.5x 20-period average and 1d ADX > 25.
Short when price breaks below 20-period Donchian low with volume confirmation and 1d ADX > 25.
Exit when price returns to the opposite Donchian band or ADX falls below 20.
This captures strong trends while avoiding choppy markets, suitable for both bull and bear regimes.
Target: 20-40 trades/year to stay within fee limits.
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
    
    # Get daily data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Daily ADX calculation for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.maximum(high_1d - low_1d, np.abs(high_1d - np.roll(close_1d, 1)))
    tr2 = np.abs(np.roll(close_1d, 1) - low_1d)
    tr = np.maximum(tr1, tr2)
    tr[0] = high_1d[0] - low_1d[0]
    
    # Directional Movement
    up_move = np.maximum(high_1d - np.roll(high_1d, 1), 0)
    down_move = np.maximum(np.roll(low_1d, 1) - low_1d, 0)
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
    
    # Align 1d ADX to 4h
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 4h Donchian channels (20-period)
    lookback = 20
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        donchian_high[i] = np.max(high[i-lookback+1:i+1])
        donchian_low[i] = np.min(low[i-lookback+1:i+1])
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 2*adx_period + 1)  # ensure ADX and Donchian are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: ADX > 25 for strong trend, < 20 to exit
        strong_trend = adx_1d_aligned[i] > 25
        weak_trend = adx_1d_aligned[i] < 20
        
        if position == 0:
            # Long: price breaks above Donchian high with volume and strong trend
            if close[i] > donchian_high[i] and vol_confirm and strong_trend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume and strong trend
            elif close[i] < donchian_low[i] and vol_confirm and strong_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to Donchian low or trend weakens
            if close[i] < donchian_low[i] or weak_trend:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to Donchian high or trend weakens
            if close[i] > donchian_high[i] or weak_trend:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_With_Volume_and_Trend_Filter_v1"
timeframe = "4h"
leverage = 1.0