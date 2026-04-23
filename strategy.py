#!/usr/bin/env python3
"""
Hypothesis: 6h strategy using 12h Elder Ray (Bull/Bear Power) with 1d ADX regime filter and volume confirmation.
Long when 12h Bull Power > 0 AND 1d ADX > 25 AND volume > 1.5x 20-period average.
Short when 12h Bear Power < 0 AND 1d ADX > 25 AND volume > 1.5x 20-period average.
Exit when Elder Power reverses sign OR ADX < 20 (regime change to ranging).
Uses discrete position sizing (0.25) to control drawdown and fee churn.
Designed for 6h timeframe to target 12-37 trades/year per symbol (50-150 total over 4 years).
Combines momentum (Elder Power), trend strength (ADX), and volume confirmation for robustness.
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
    
    # Calculate 12h Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    ema_13_12h = pd.Series(df_12h['close'].values).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = df_12h['high'].values - ema_13_12h
    bear_power = df_12h['low'].values - ema_13_12h
    
    # Align 12h Elder Power to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_12h, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_12h, bear_power)
    
    # Calculate 1d ADX(14) for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr1[0]
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = -np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed TR, +DM, -DM
    tr_14 = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_dm_14 = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_14 = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_14 / tr_14
    minus_di = 100 * minus_dm_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume average (20-period) on 6h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 14, 20)  # Elder Power needs 13, ADX needs 14, vol MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        bull_val = bull_power_aligned[i]
        bear_val = bear_power_aligned[i]
        adx_val = adx_aligned[i]
        
        if position == 0:
            # Long: Bull Power positive AND ADX > 25 (strong trend) AND volume spike
            if (bull_val > 0 and adx_val > 25 and volume[i] > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power negative AND ADX > 25 (strong trend) AND volume spike
            elif (bear_val < 0 and adx_val > 25 and volume[i] > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Elder Power reverses sign
            if position == 1 and bull_val <= 0:
                exit_signal = True
            elif position == -1 and bear_val >= 0:
                exit_signal = True
            
            # Secondary exit: ADX < 20 (regime change to ranging)
            if adx_val < 20:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_ElderRay_ADX25_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0