#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using daily Vortex indicator with volume confirmation and ADX filter
# Vortex identifies trend direction (VI+ > VI- = uptrend, VI- > VI+ = downtrend)
# Works in both bull/bear markets: trend following in trending regimes, avoids whipsaws in ranging markets
# Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "12h_Vortex_ADX_VolumeFilter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate daily Vortex and ADX ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:  # Need enough data for ADX(14) and Vortex(14)
        return np.zeros(n)
    
    # Calculate True Range for Vortex and ADX
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Vortex Indicator (14-period)
    vm_plus = abs(df_1d['high'] - df_1d['low'].shift(1))
    vm_minus = abs(df_1d['low'] - df_1d['high'].shift(1))
    
    # Sum over 14 periods
    period = 14
    sum_vm_plus = vm_plus.rolling(window=period, min_periods=period).sum()
    sum_vm_minus = vm_minus.rolling(window=period, min_periods=period).sum()
    sum_tr = tr.rolling(window=period, min_periods=period).sum()
    
    vi_plus = sum_vm_plus / sum_tr
    vi_minus = sum_vm_minus / sum_tr
    
    # ADX (14-period)
    # Calculate directional movement
    dm_plus = np.where((df_1d['high'] - df_1d['high'].shift(1)) > (df_1d['low'].shift(1) - df_1d['low']), 
                       np.maximum(df_1d['high'] - df_1d['high'].shift(1), 0), 0)
    dm_minus = np.where((df_1d['low'].shift(1) - df_1d['low']) > (df_1d['high'] - df_1d['high'].shift(1)), 
                        np.maximum(df_1d['low'].shift(1) - df_1d['low'], 0), 0)
    
    # Smoothed DM and TR
    dm_plus_smooth = pd.Series(dm_plus).rolling(window=period, min_periods=period).sum()
    dm_minus_smooth = pd.Series(dm_minus).rolling(window=period, min_periods=period).sum()
    tr_smooth = tr.rolling(window=period, min_periods=period).sum()
    
    # DI+ and DI-
    di_plus = 100 * (dm_plus_smooth / tr_smooth)
    di_minus = 100 * (dm_minus_smooth / tr_smooth)
    
    # DX and ADX
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = dx.rolling(window=period, min_periods=period).mean()
    
    # Convert to numpy arrays
    vi_plus_arr = vi_plus.values
    vi_minus_arr = vi_minus.values
    adx_arr = adx.values
    
    # Align daily indicators to 12h timeframe
    vi_plus_aligned = align_htf_to_ltf(prices, df_1d, vi_plus_arr)
    vi_minus_aligned = align_htf_to_ltf(prices, df_1d, vi_minus_arr)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_arr)
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(vi_plus_aligned[i]) or np.isnan(vi_minus_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # ADX filter: only trade when ADX > 20 (trending market)
        adx_filter = adx_aligned[i] > 20
        
        if position == 0:
            # Long entry: VI+ > VI- (uptrend) with volume confirmation
            if vi_plus_aligned[i] > vi_minus_aligned[i] and volume_filter[i] and adx_filter:
                signals[i] = 0.25
                position = 1
            # Short entry: VI- > VI+ (downtrend) with volume confirmation
            elif vi_minus_aligned[i] > vi_plus_aligned[i] and volume_filter[i] and adx_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: VI- crosses above VI+ (trend reversal) or ADX weakens
            if vi_minus_aligned[i] > vi_plus_aligned[i] or adx_aligned[i] < 18:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: VI+ crosses above VI- (trend reversal) or ADX weakens
            if vi_plus_aligned[i] > vi_minus_aligned[i] or adx_aligned[i] < 18:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals