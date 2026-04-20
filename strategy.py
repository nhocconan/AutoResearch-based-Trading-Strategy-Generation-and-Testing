#!/usr/bin/env python3
# 4h_Camarilla_Pivot_R1S1_Breakout_Volume_Confirmation
# Hypothesis: Price breaking above/below Camarilla R1/S1 levels from 1d, confirmed by volume spike and ADX trend filter.
# Works in bull/bear: Uses Camarilla for reversal/breakout signals, volume to confirm strength, ADX to avoid chop.
# Target: 20-40 trades/year (80-160 total over 4 years) to avoid fee drag.

name = "4h_Camarilla_Pivot_R1S1_Breakout_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 1d bar: R1, S1
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla: R1 = close + 1.1/12 * (high - low), S1 = close - 1.1/12 * (high - low)
    camarilla_R1 = close_1d + (1.1/12) * (high_1d - low_1d)
    camarilla_S1 = close_1d - (1.1/12) * (high_1d - low_1d)
    
    # Align to 4h timeframe (wait for 1d bar to close)
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    
    # Calculate volume spike: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Calculate ADX(14) for trend filter
    period = 14
    
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
    
    # Smoothed values using Wilder's smoothing (alpha = 1/period)
    atr = np.full_like(high, np.nan)
    dm_plus_smooth = np.full_like(high, np.nan)
    dm_minus_smooth = np.full_like(high, np.nan)
    
    # Initial values (simple average of first 'period' values)
    if len(high) >= period:
        atr[period] = np.nanmean(tr[1:period+1])
        dm_plus_smooth[period] = np.nanmean(dm_plus[1:period+1])
        dm_minus_smooth[period] = np.nanmean(dm_minus[1:period+1])
        
        # Wilder's smoothing for subsequent values
        for i in range(period + 1, len(high)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period - 1) + dm_plus[i]) / period
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period - 1) + dm_minus[i]) / period
    
    # DI and DX
    di_plus = np.full_like(high, np.nan)
    di_minus = np.full_like(high, np.nan)
    dx = np.full_like(high, np.nan)
    
    valid = ~np.isnan(atr) & (atr != 0)
    di_plus[valid] = (dm_plus_smooth[valid] / atr[valid]) * 100
    di_minus[valid] = (dm_minus_smooth[valid] / atr[valid]) * 100
    
    dx_valid = valid & ~np.isnan(di_plus) & ~np.isnan(di_minus) & ((di_plus + di_minus) != 0)
    dx[dx_valid] = (np.abs(di_plus[dx_valid] - di_minus[dx_valid]) / (di_plus[dx_valid] + di_minus[dx_valid])) * 100
    
    # ADX (smoothed DX)
    adx = np.full_like(high, np.nan)
    if len(high) >= 2 * period:
        adx[2*period] = np.nanmean(dx[period+1:2*period+1])
        for i in range(2*period + 1, len(high)):
            adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(2*period + 1, 50)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_R1_aligned[i]) or np.isnan(camarilla_S1_aligned[i]) or 
            np.isnan(adx[i]) or np.isnan(di_plus[i]) or np.isnan(di_minus[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 + volume spike + ADX > 20 (trending)
            if close[i] > camarilla_R1_aligned[i] and volume_spike[i] and adx[i] > 20:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + volume spike + ADX > 20 (trending)
            elif close[i] < camarilla_S1_aligned[i] and volume_spike[i] and adx[i] > 20:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below S1 (reversal) or ADX weakens
            if close[i] < camarilla_S1_aligned[i] or adx[i] < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above R1 (reversal) or ADX weakens
            if close[i] > camarilla_R1_aligned[i] or adx[i] < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals