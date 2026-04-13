#!/usr/bin/env python3
"""
12h_1d_Camarilla_Pivot_Breakout_Volume
Hypothesis: Uses Camarilla pivot levels from daily timeframe on 12h chart. Enters long when price breaks above H4 level with volume > 1.5x average, short when breaks below L4 level. Uses 12h ADX > 25 to filter trending markets. Works in both bull and bear by trading momentum after range breakouts. Target: 15-35 trades/year on 12h (60-140 total over 4 years).
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
    if len(df_1d) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each day
    # H4 = close + 1.1 * (high - low) / 2
    # L4 = close - 1.1 * (high - low) / 2
    # H3 = close + 1.1 * (high - low) / 4
    # L3 = close - 1.1 * (high - low) / 4
    # H2 = close + 1.1 * (high - low) / 6
    # L2 = close - 1.1 * (high - low) / 6
    # H1 = close + 1.1 * (high - low) / 12
    # L1 = close - 1.1 * (high - low) / 12
    
    range_1d = high_1d - low_1d
    H4 = close_1d + 1.1 * range_1d / 2
    L4 = close_1d - 1.1 * range_1d / 2
    H3 = close_1d + 1.1 * range_1d / 4
    L3 = close_1d - 1.1 * range_1d / 4
    H2 = close_1d + 1.1 * range_1d / 6
    L2 = close_1d - 1.1 * range_1d / 6
    H1 = close_1d + 1.1 * range_1d / 12
    L1 = close_1d - 1.1 * range_1d / 12
    
    # Align Camarilla levels to 12h timeframe
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    H2_aligned = align_htf_to_ltf(prices, df_1d, H2)
    L2_aligned = align_htf_to_ltf(prices, df_1d, L2)
    H1_aligned = align_htf_to_ltf(prices, df_1d, H1)
    L1_aligned = align_htf_to_ltf(prices, df_1d, L1)
    
    # Calculate 12h ADX for trend filtering
    # ADX requires +DI and -DI calculation
    # +DI = 100 * EMA(|+DM|, 14) / ATR
    # -DI = 100 * EMA(|-DM|, 14) / ATR
    # ADX = EMA(|+DI - -DI| / (+DI + -DI), 14)
    
    # Calculate True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Calculate +DM and -DM
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Calculate ATR and DX
    atr = np.zeros(n)
    atr[0] = np.nan
    for i in range(1, n):
        if i < 14:
            atr[i] = np.nan
        else:
            if np.isnan(atr[i-1]):
                atr[i] = np.nanmean(tr[i-13:i+1])
            else:
                atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Calculate +DI and -DI
    di_plus = np.zeros(n)
    di_minus = np.zeros(n)
    for i in range(14, n):
        if np.isnan(atr[i]) or atr[i] == 0:
            di_plus[i] = 0
            di_minus[i] = 0
        else:
            # Calculate smoothed +DM and -DM
            dm_plus_smooth = np.zeros(n)
            dm_minus_smooth = np.zeros(n)
            if i == 14:
                dm_plus_smooth[i] = np.nansum(dm_plus[i-13:i+1])
                dm_minus_smooth[i] = np.nansum(dm_minus[i-13:i+1])
            else:
                dm_plus_smooth[i] = dm_plus_smooth[i-1] - (dm_plus_smooth[i-1] / 14) + dm_plus[i]
                dm_minus_smooth[i] = dm_minus_smooth[i-1] - (dm_minus_smooth[i-1] / 14) + dm_minus[i]
            
            di_plus[i] = 100 * dm_plus_smooth[i] / atr[i] if atr[i] != 0 else 0
            di_minus[i] = 100 * dm_minus_smooth[i] / atr[i] if atr[i] != 0 else 0
    
    # Calculate ADX
    dx = np.zeros(n)
    adx = np.zeros(n)
    for i in range(14, n):
        if di_plus[i] + di_minus[i] == 0:
            dx[i] = 0
        else:
            dx[i] = 100 * np.abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
    
    for i in range(28, n):
        if i < 28:
            adx[i] = np.nan
        else:
            if np.isnan(adx[i-1]):
                adx[i] = np.nanmean(dx[i-13:i+1])
            else:
                adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Volume confirmation: 20-period average
    vol_ma_20 = np.zeros(n)
    vol_series = pd.Series(volume)
    for i in range(n):
        if i < 20:
            vol_ma_20[i] = np.nan
        else:
            vol_ma_20[i] = vol_series.iloc[i-19:i+1].mean()
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(30, n):
        # Skip if not in session or data not ready
        if not session_mask[i] or \
           np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]) or \
           np.isnan(adx[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # Trend filter: only trade when ADX > 25 (trending market)
        if adx[i] <= 25:
            # No strong trend - stay flat
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_expansion = volume[i] > (vol_ma_20[i] * 1.5) if not np.isnan(vol_ma_20[i]) else False
        
        # Breakout conditions
        long_breakout = (close[i] > H4_aligned[i]) and volume_expansion
        short_breakout = (close[i] < L4_aligned[i]) and volume_expansion
        
        # Entry logic
        if long_breakout and position != 1:
            position = 1
            signals[i] = position_size
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1:
            # Hold long position
            signals[i] = position_size
        elif position == -1:
            # Hold short position
            signals[i] = -position_size
        else:
            # Flat
            signals[i] = 0.0
    
    return signals

name = "12h_1d_Camarilla_Pivot_Breakout_Volume"
timeframe = "12h"
leverage = 1.0