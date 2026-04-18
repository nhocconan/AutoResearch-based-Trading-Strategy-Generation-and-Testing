#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_Volume_Regime_v1
Hypothesis: Use daily Camarilla pivot levels (R1/S1) for support/resistance on 12h timeframe. 
Go long when price breaks above R1 with volume confirmation and chop regime filtering (trending market). 
Go short when price breaks below S1 with volume confirmation and chop regime filtering.
Uses 1-week ADX for trend strength confirmation to avoid choppy markets.
Target: 15-30 trades/year by requiring multiple confluence factors (pivot break, volume, trend).
Works in bull via R1 breakouts and in bear via S1 breakdowns.
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
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each day
    # Pivot = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = close_1d + (high_1d - low_1d) * 1.1 / 12.0
    s1_1d = close_1d - (high_1d - low_1d) * 1.1 / 12.0
    
    # Align Camarilla levels to 12h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Get 1w data for ADX (trend strength)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX(14) on weekly data
    def calculate_adx(high, low, close, period=14):
        n = len(high)
        if n < period * 2:
            return np.full(n, np.nan)
        
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm = np.concatenate([[0.0], plus_dm])
        minus_dm = np.concatenate([[0.0], minus_dm])
        
        # Smoothed values
        atr = np.full(n, np.nan)
        plus_di = np.full(n, np.nan)
        minus_di = np.full(n, np.nan)
        
        # Initial averages
        if n >= period + 1:
            atr[period] = np.nanmean(tr[1:period+1])
            plus_dm_sum = np.nansum(plus_dm[1:period+1])
            minus_dm_sum = np.nansum(minus_dm[1:period+1])
            
            if atr[period] != 0:
                plus_di[period] = (plus_dm_sum / atr[period]) * 100
                minus_di[period] = (minus_dm_sum / atr[period]) * 100
            
            # Wilder smoothing
            for i in range(period + 1, n):
                atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
                plus_di[i] = (plus_di[i-1] * (period - 1) + plus_dm[i]) / period * 100 / (atr[i] if atr[i] != 0 else 1)
                minus_di[i] = (minus_di[i-1] * (period - 1) + minus_dm[i]) / period * 100 / (atr[i] if atr[i] != 0 else 1)
        
        # Calculate DX and ADX
        dx = np.full(n, np.nan)
        adx = np.full(n, np.nan)
        
        for i in range(period, n):
            if plus_di[i] + minus_di[i] != 0:
                dx[i] = np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i]) * 100
        
        # ADX is smoothed DX
        if n >= 2 * period:
            adx[2*period-1] = np.nanmean(dx[period:2*period])
            for i in range(2*period, n):
                adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
        
        return adx
    
    adx_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(vol_period, 1) + 1  # Need volume MA and aligned data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(adx_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_1w_aligned[i] > 25
        
        if position == 0 and trending:
            # Long: price breaks above R1 with volume confirmation
            if close[i] > r1_1d_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume confirmation
            elif close[i] < s1_1d_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S1 or ADX drops below 20 (losing trend)
            if close[i] < s1_1d_aligned[i] or adx_1w_aligned[i] < 20:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1 or ADX drops below 20 (losing trend)
            if close[i] > r1_1d_aligned[i] or adx_1w_aligned[i] < 20:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_Volume_Regime_v1"
timeframe = "12h"
leverage = 1.0