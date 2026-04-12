#!/usr/bin/env python3
"""
4h_1d_camarilla_breakout_volume_v4
Hypothesis: Refined 4-hour strategy using daily Camarilla pivot levels with volume confirmation and ADX trend filter.
Enters long when price breaks above H3 with volume spike and ADX > 20 (trending market).
Short when breaks below L3 with volume spike and ADX > 20.
Uses discrete position sizing (0.25) to minimize churn and exits on opposite H3/L3 touch.
Target: 20-30 trades/year (80-120 total over 4 years) to minimize fee drag while capturing strong trends.
Focus on breakout quality with volume and trend filters to avoid false signals in chop.
"""

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
    
    # Get daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily data needed for indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values for pivot calculation (no look-ahead)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla calculations using previous day's data
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # Camarilla levels
    h3 = pivot + 1.1 * range_val / 2
    l3 = pivot - 1.1 * range_val / 2
    h4 = pivot + 1.1 * range_val
    l4 = pivot - 1.1 * range_val
    
    # Align Camarilla levels to 4h timeframe
    h3_4h = align_htf_to_ltf(prices, df_1d, h3)
    l3_4h = align_htf_to_ltf(prices, df_1d, l3)
    h4_4h = align_htf_to_ltf(prices, df_1d, h4)
    l4_4h = align_htf_to_ltf(prices, df_1d, l4)
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Calculate ADX (14-period) on 4h data for trend filter
    # +DM, -DM, TR
    up_move = np.diff(high, prepend=high[0])
    down_move = np.diff(low, prepend=low[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = np.maximum(np.abs(np.diff(high, prepend=high[0])), 
                    np.maximum(np.abs(np.diff(low, prepend=low[0])), 
                               np.abs(np.diff(close, prepend=close[0]))))
    
    # Smoothed values (using Wilder's smoothing)
    atr = np.full(n, np.nan)
    plus_dm_smooth = np.full(n, np.nan)
    minus_dm_smooth = np.full(n, np.nan)
    
    # Initialize first values
    atr[13] = np.mean(tr[1:14]) if n > 13 else np.nan
    plus_dm_smooth[13] = np.mean(plus_dm[1:14]) if n > 13 else np.nan
    minus_dm_smooth[13] = np.mean(minus_dm[1:14]) if n > 13 else np.nan
    
    # Wilder's smoothing
    for i in range(14, n):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
        plus_dm_smooth[i] = (plus_dm_smooth[i-1] * 13 + plus_dm[i]) / 14
        minus_dm_smooth[i] = (minus_dm_smooth[i-1] * 13 + minus_dm[i]) / 14
    
    # Calculate DI and DX
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    dx = np.full(n, np.nan)
    
    for i in range(14, n):
        if atr[i] != 0:
            plus_di[i] = (plus_dm_smooth[i] / atr[i]) * 100
            minus_di[i] = (minus_dm_smooth[i] / atr[i]) * 100
            if (plus_di[i] + minus_di[i]) != 0:
                dx[i] = (np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100
    
    # Calculate ADX (smoothed DX)
    adx = np.full(n, np.nan)
    if n > 27:
        adx[27] = np.mean(dx[14:28]) if not np.any(np.isnan(dx[14:28])) else np.nan
        for i in range(28, n):
            if not np.isnan(dx[i]):
                adx[i] = (adx[i-1] * 13 + dx[i]) / 14
            else:
                adx[i] = adx[i-1]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(h3_4h[i]) or np.isnan(l3_4h[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.8x 20-period average
        volume_filter = volume[i] > vol_ma[i] * 1.8
        
        # Trend filter: ADX > 20 (trending market)
        trend_filter = adx[i] > 20
        
        # Entry conditions: Camarilla H3/L3 breakout with volume and trend confirmation
        long_breakout = close[i] > h3_4h[i] and volume_filter and trend_filter
        short_breakout = close[i] < l3_4h[i] and volume_filter and trend_filter
        
        # Exit conditions: touch opposite H3/L3 level
        long_exit = close[i] < l3_4h[i]
        short_exit = close[i] > h3_4h[i]
        
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_camarilla_breakout_volume_v4"
timeframe = "4h"
leverage = 1.0