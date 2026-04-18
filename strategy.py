#!/usr/bin/env python3
"""
12h_Donchian_20_Breakout_Volume_Confirm_v2
Hypothesis: Combines 12h Donchian(20) breakout with 1d volume confirmation and ADX trend filter.
Enters long on upper band breakout with volume > 1.5x 20-period average and ADX > 20.
Enters short on lower band breakout with volume spike and ADX > 20.
Designed for low-moderate trade frequency (~15-25/year) with trend-following capability in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian(20) channels
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(19, n):
        upper[i] = np.max(high[i-19:i+1])
        lower[i] = np.min(low[i-19:i+1])
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    # ADX(14) calculation
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    atr = np.full(n, np.nan)
    dm_plus_smooth = np.full(n, np.nan)
    dm_minus_smooth = np.full(n, np.nan)
    
    for i in range(14, n):
        if i == 14:
            atr[i] = np.mean(tr[1:15])
            dm_plus_smooth[i] = np.mean(dm_plus[1:15])
            dm_minus_smooth[i] = np.mean(dm_minus[1:15])
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * 13 + dm_plus[i]) / 14
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * 13 + dm_minus[i]) / 14
    
    # DI+ and DI-
    di_plus = np.full(n, np.nan)
    di_minus = np.full(n, np.nan)
    for i in range(14, n):
        if atr[i] != 0:
            di_plus[i] = 100 * dm_plus_smooth[i] / atr[i]
            di_minus[i] = 100 * dm_minus_smooth[i] / atr[i]
        else:
            di_plus[i] = 0
            di_minus[i] = 0
    
    # DX and ADX
    dx = np.full(n, np.nan)
    for i in range(14, n):
        if di_plus[i] + di_minus[i] != 0:
            dx[i] = 100 * np.abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
        else:
            dx[i] = 0
    
    adx = np.full(n, np.nan)
    for i in range(28, n):  # 14 + 14 for smoothing
        if i == 28:
            adx[i] = np.mean(dx[15:29])
        else:
            adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Warmup
    
    for i in range(start_idx, n):
        if np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(vol_ma[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Upper band breakout with volume spike and ADX > 20
            if close[i] > upper[i] and vol_spike[i] and adx[i] > 20:
                signals[i] = 0.25
                position = 1
            # Short: Lower band breakout with volume spike and ADX > 20
            elif close[i] < lower[i] and vol_spike[i] and adx[i] > 20:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Close below lower band or ADX drops below 15
            if close[i] < lower[i] or adx[i] < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Close above upper band or ADX drops below 15
            if close[i] > upper[i] or adx[i] < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian_20_Breakout_Volume_Confirm_v2"
timeframe = "12h"
leverage = 1.0