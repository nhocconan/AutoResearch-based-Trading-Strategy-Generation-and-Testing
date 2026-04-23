#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray + 1d ADX regime + volume confirmation.
Long when Bull Power > 0 AND Bear Power < 0 AND 1d ADX > 25 (trending) AND volume > 1.5x 20-period average.
Short when Bear Power > 0 AND Bull Power < 0 AND 1d ADX > 25 AND volume > 1.5x 20-period average.
Exit when Elder Power signals reverse OR ADX < 20 (range) OR volume drops.
Uses 1d HTF for ADX regime filter (avoids whipsaws in ranging markets). Target: 50-150 total trades over 4 years (12-37/year).
Elder Ray captures bull/bear strength via EMA13; ADX filters for trending conditions only.
Works in both bull (trend continuation) and bear (strong downtrends) markets.
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
    
    # Calculate 1d ADX for regime filter (HTF)
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
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values (Wilder's smoothing = EMA with alpha=1/period)
    period = 14
    alpha = 1.0 / period
    tr_period = pd.Series(tr).ewm(alpha=alpha, adjust=False, min_periods=period).mean().values
    dm_plus_period = pd.Series(dm_plus).ewm(alpha=alpha, adjust=False, min_periods=period).mean().values
    dm_minus_period = pd.Series(dm_minus).ewm(alpha=alpha, adjust=False, min_periods=period).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_period / tr_period
    di_minus = 100 * dm_minus_period / tr_period
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=alpha, adjust=False, min_periods=period).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13) on 6h
    ema_period = 13
    ema_13 = pd.Series(close).ewm(span=ema_period, adjust=False, min_periods=ema_period).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # 20-period volume average for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(ema_period, 20, 30)  # Elder Ray (13), volume MA (20), ADX (30)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_val = adx_aligned[i]
        bull = bull_power[i]
        bear = bear_power[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power < 0 AND ADX > 25 (trending) AND volume spike
            if bull > 0 and bear < 0 and adx_val > 25 and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power > 0 AND Bull Power < 0 AND ADX > 25 (trending) AND volume spike
            elif bear > 0 and bull < 0 and adx_val > 25 and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Bull Power <= 0 OR Bear Power >= 0 OR ADX < 20 (range) OR volume drop
                if bull <= 0 or bear >= 0 or adx_val < 20 or volume[i] < vol_ma_val:
                    exit_signal = True
            elif position == -1:
                # Short exit: Bear Power <= 0 OR Bull Power >= 0 OR ADX < 20 (range) OR volume drop
                if bear <= 0 or bull >= 0 or adx_val < 20 or volume[i] < vol_ma_val:
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