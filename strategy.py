#!/usr/bin/env python3
"""
6h_Pivot_R1_S1_Breakout_Volume_ADXFilter_v1
Hypothesis: Trade daily pivot breakouts on 6h with volume and ADX confirmation. 
Enter long when price breaks above R1 with volume > 1.5x average and ADX > 20 (trending). 
Enter short when price breaks below S1 with volume > 1.5x average and ADX > 20. 
Exit when price crosses the daily pivot point (PP) or ADX < 15 (range). 
Uses daily pivot levels calculated from prior day's OHLC. 
Works in bull/bear by following institutional pivot levels with trend filter. 
Targets 15-25 trades/year via strict breakout conditions + volume + ADX filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily pivot points (PP, R1, S1) from prior day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot Point = (H + L + C) / 3
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    # R1 = 2*PP - L
    r1_1d = 2 * pp_1d - low_1d
    # S1 = 2*PP - H
    s1_1d = 2 * pp_1d - high_1d
    
    # Align daily pivot levels to 6h timeframe (use prior day's levels)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # ADX(14) on 6h
    period = 14
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # First TR
    
    # Directional Movement
    up_move = np.diff(high, prepend=high[0])
    down_move = -np.diff(low, prepend=low[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    atr = np.zeros_like(tr)
    plus_di = np.zeros_like(tr)
    minus_di = np.zeros_like(tr)
    
    if len(tr) >= period:
        # Initial averages
        atr[period-1] = np.mean(tr[:period])
        plus_dm_sum = np.sum(plus_dm[:period])
        minus_dm_sum = np.sum(minus_dm[:period])
        
        # Wilder smoothing
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
            plus_dm_sum = plus_dm_sum - (plus_dm_sum / period) + plus_dm[i]
            minus_dm_sum = minus_dm_sum - (minus_dm_sum / period) + minus_dm[i]
            plus_di[i] = 100 * plus_dm_sum / atr[i] if atr[i] != 0 else 0
            minus_di[i] = 100 * minus_dm_sum / atr[i] if atr[i] != 0 else 0
        
        # DX and ADX
        dx = np.zeros_like(tr)
        for i in range(period, len(tr)):
            if plus_di[i] + minus_di[i] != 0:
                dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        adx = np.zeros_like(tr)
        if len(tr) >= 2 * period - 1:
            adx[2*period-2] = np.mean(dx[period-1:2*period-1])
            for i in range(2*period-1, len(tr)):
                adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    else:
        adx = np.zeros_like(tr)
    
    # Volume confirmation: volume > 1.5x 24-period average
    vol_ma = np.zeros_like(volume)
    vol_period = 24
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    else:
        vol_ma[:] = np.mean(volume) if len(volume) > 0 else 0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, vol_period, 2*period-1)  # Ensure ADX and vol MA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(adx[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: break above R1 + volume + ADX > 20
            if close[i] > r1_aligned[i] and vol_confirm and adx[i] > 20:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 + volume + ADX > 20
            elif close[i] < s1_aligned[i] and vol_confirm and adx[i] > 20:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: cross below PP or ADX < 15 (range)
            if close[i] < pp_aligned[i] or adx[i] < 15:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: cross above PP or ADX < 15 (range)
            if close[i] > pp_aligned[i] or adx[i] < 15:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Pivot_R1_S1_Breakout_Volume_ADXFilter_v1"
timeframe = "6h"
leverage = 1.0