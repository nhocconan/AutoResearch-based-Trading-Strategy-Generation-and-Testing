#!/usr/bin/env python3
"""
1d_Camarilla_Pivot_R1S1_Breakout_Volume
Hypothesis: Use weekly Camarilla pivot levels (R1, S1) from the previous week for breakout signals on daily timeframe. Go long when price breaks above weekly R1 with volume confirmation, short when price breaks below weekly S1 with volume confirmation. Uses weekly ADX > 25 to filter for trending markets only. Designed to work in both bull (breakouts) and bear (breakdowns) markets with low trade frequency to minimize fee drag.
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
    
    # Get weekly data for Camarilla pivot levels
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla pivot levels for weekly timeframe
    # Camarilla formulas: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    r1_1w = np.full_like(close_1w, np.nan)
    s1_1w = np.full_like(close_1w, np.nan)
    
    for i in range(len(close_1w)):
        if not (np.isnan(high_1w[i]) or np.isnan(low_1w[i]) or np.isnan(close_1w[i])):
            r1_1w[i] = close_1w[i] + (high_1w[i] - low_1w[i]) * 1.1 / 12
            s1_1w[i] = close_1w[i] - (high_1w[i] - low_1w[i]) * 1.1 / 12
    
    # Calculate weekly ADX for trend filtering
    # ADX calculation requires +DI and -DI
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # align with original arrays
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        plus_dm = np.concatenate([[np.nan], plus_dm])
        minus_dm = np.concatenate([[np.nan], minus_dm])
        
        # Smoothed values
        atr = np.full_like(tr, np.nan)
        plus_di = np.full_like(tr, np.nan)
        minus_di = np.full_like(tr, np.nan)
        
        if len(tr) >= period:
            # Initial values
            atr[period] = np.nanmean(tr[1:period+1])
            plus_dm_sum = np.nansum(plus_dm[1:period+1])
            minus_dm_sum = np.nansum(minus_dm[1:period+1])
            
            plus_di[period] = 100 * plus_dm_sum / (atr[period] * period) if atr[period] > 0 else 0
            minus_di[period] = 100 * minus_dm_sum / (atr[period] * period) if atr[period] > 0 else 0
            
            # Wilder smoothing
            for i in range(period+1, len(tr)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
                plus_dm_val = plus_dm[i] if not np.isnan(plus_dm[i]) else 0
                minus_dm_val = minus_dm[i] if not np.isnan(minus_dm[i]) else 0
                plus_di[i] = 100 * (plus_di[i-1] * (period-1) + plus_dm_val) / (atr[i] * period) if atr[i] > 0 else 0
                minus_di[i] = 100 * (minus_di[i-1] * (period-1) + minus_dm_val) / (atr[i] * period) if atr[i] > 0 else 0
        
        # DX and ADX
        dx = np.full_like(tr, np.nan)
        adx = np.full_like(tr, np.nan)
        
        for i in range(period, len(tr)):
            if plus_di[i] + minus_di[i] > 0:
                dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        if len(tr) >= 2*period:
            adx[2*period-1] = np.nanmean(dx[period:2*period])
            for i in range(2*period, len(tr)):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    
    # Align weekly data to daily timeframe
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Volume confirmation: volume > 1.5x 20-day average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(vol_period, 28) + 1  # ensure ADX and volume data available
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) or 
            np.isnan(adx_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Only trade in trending markets (ADX > 25)
        trending = adx_1w_aligned[i] > 25
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0 and trending:
            # Long: price breaks above weekly R1 + volume
            if close[i] > r1_1w_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S1 + volume
            elif close[i] < s1_1w_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below weekly S1
            if close[i] < s1_1w_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above weekly R1
            if close[i] > r1_1w_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_Pivot_R1S1_Breakout_Volume"
timeframe = "1d"
leverage = 1.0