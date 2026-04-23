#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Power with 1d ADX regime filter and volume confirmation.
Long when Bull Power > 0 AND Bear Power < 0 AND 1d ADX > 25 (trending) AND volume > 1.5x 20-period average.
Short when Bear Power < 0 AND Bull Power > 0 AND 1d ADX > 25 (trending) AND volume > 1.5x 20-period average.
Exit when power signals reverse or ADX < 20 (range regime).
Uses 1d HTF for ADX trend strength filter (avoids whipsaws in ranging markets). Target: 50-150 total trades over 4 years (12-37/year).
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
    
    # Calculate 1d ADX for trend strength filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values (Wilder's smoothing = EMA with alpha=1/period)
    period = 14
    alpha = 1.0 / period
    tr_smooth = pd.Series(tr).ewm(alpha=alpha, adjust=False).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=alpha, adjust=False).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=alpha, adjust=False).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=alpha, adjust=False).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Elder Ray Power (13-period EMA as reference)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # 20-period volume average for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 30, 20)  # Elder Ray (13), ADX (30), volume MA (20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bp = bull_power[i]
        br = bear_power[i]
        adx_val = adx_aligned[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power < 0 AND ADX > 25 (trending) AND volume spike
            if bp > 0 and br < 0 and adx_val > 25 and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND Bull Power > 0 AND ADX > 25 (trending) AND volume spike
            elif br < 0 and bp > 0 and adx_val > 25 and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Bull Power <= 0 OR Bear Power >= 0 OR ADX < 20 (range)
                if bp <= 0 or br >= 0 or adx_val < 20:
                    exit_signal = True
            elif position == -1:
                # Short exit: Bear Power >= 0 OR Bull Power <= 0 OR ADX < 20 (range)
                if br >= 0 or bp <= 0 or adx_val < 20:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_ElderRay_Power_1dADX_Regime_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0