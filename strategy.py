#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d ADX trend filter + volume confirmation
# Donchian breakout captures momentum in both bull/bear markets
# 1d ADX > 25 ensures we only trade in strong trending regimes (avoids chop)
# Volume confirmation: current volume > 2.0x 20-period MA to confirm breakout strength
# Timeframe: 12h, HTF: 1d. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.
# Uses discrete position sizing (0.25) to minimize fee churn and control drawdown.

name = "12h_Donchian20_1dADX_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for ADX and Donchian
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # need sufficient data for ADX and Donchian
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for ADX
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1d ADX(14)
    if len(close_1d) >= 14:
        # Directional Movement
        dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                           np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
        dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                            np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
        dm_plus = np.concatenate([[0], dm_plus])
        dm_minus = np.concatenate([[0], dm_minus])
        
        # Smoothed values
        dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
        dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
        atr_14_smooth = pd.Series(atr_14).ewm(span=14, adjust=False, min_periods=14).mean().values
        
        # Directional Indicators
        di_plus = 100 * dm_plus_smooth / np.where(atr_14_smooth == 0, np.nan, atr_14_smooth)
        di_minus = 100 * dm_minus_smooth / np.where(atr_14_smooth == 0, np.nan, atr_14_smooth)
        
        # DX and ADX
        dx = 100 * np.abs(di_plus - di_minus) / np.where((di_plus + di_minus) == 0, np.nan, (di_plus + di_minus))
        adx_1d = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    else:
        adx_1d = np.full(len(close_1d), np.nan)
    
    # Calculate 1d Donchian(20) channels
    if len(close_1d) >= 20:
        donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
        donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    else:
        donchian_high = np.full(len(close_1d), np.nan)
        donchian_low = np.full(len(close_1d), np.nan)
    
    # Align 1d indicators to 12h
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Volume confirmation on 12h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (2.0 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: ADX>25 AND price breaks above Donchian high AND volume spike
            if (adx_1d_aligned[i] > 25 and 
                close[i] > donchian_high_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: ADX>25 AND price breaks below Donchian low AND volume spike
            elif (adx_1d_aligned[i] > 25 and 
                  close[i] < donchian_low_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below Donchian low (breakdown) OR ADX < 20 (trend weakening)
            if close[i] < donchian_low_aligned[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Donchian high (breakout) OR ADX < 20 (trend weakening)
            if close[i] > donchian_high_aligned[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals