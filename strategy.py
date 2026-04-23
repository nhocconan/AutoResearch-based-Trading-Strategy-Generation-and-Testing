#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Bull/Bear Power with 1d ADX regime filter and volume confirmation.
Long when Bull Power > 0 AND Bear Power < 0 AND 1d ADX > 25 (trending) AND volume > 1.5x 20-period average.
Short when Bear Power < 0 AND Bull Power > 0 AND 1d ADX > 25 (trending) AND volume > 1.5x 20-period average.
Exit when ADX < 20 (range) or power diverges.
Uses 1d HTF for ADX regime (avoids whipsaws in ranging markets). Target: 50-150 total trades over 4 years (12-37/year).
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
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align length
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smooth TR, +DM, -DM (Wilder smoothing = EMA with alpha=1/period)
    def WilderSmooth(data, period):
        data = np.asarray(data)
        length = len(data)
        result = np.full(length, np.nan)
        if length < period:
            return result
        # first value: simple average
        result[period-1] = np.nanmean(data[:period])
        # rest: Wilder smoothing
        for i in range(period, length):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period = 14
    tr_smooth = WilderSmooth(tr, period)
    plus_dm_smooth = WilderSmooth(plus_dm, period)
    minus_dm_smooth = WilderSmooth(minus_dm, period)
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # DX and ADX
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    adx = WilderSmooth(dx, period)
    adx_1d = adx  # already aligned to 1d
    
    # Align ADX to 6h
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate Elder Ray Power (6h)
    # Bull Power = High - EMA(13)
    # Bear Power = Low - EMA(13)
    ema_period = 13
    ema_13 = pd.Series(close).ewm(span=ema_period, adjust=False, min_periods=ema_period).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # 20-period volume average for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(ema_period, 20, 30)  # EMA13, volume MA, ADX buffers
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_val = adx_1d_aligned[i]
        bull = bull_power[i]
        bear = bear_power[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power < 0 AND ADX > 25 (trending) AND volume spike
            if bull > 0 and bear < 0 and adx_val > 25 and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND Bull Power > 0 AND ADX > 25 (trending) AND volume spike
            elif bear < 0 and bull > 0 and adx_val > 25 and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: ADX < 20 (range) OR Bear Power >= 0 (bullish momentum fading)
                if adx_val < 20 or bear >= 0:
                    exit_signal = True
            elif position == -1:
                # Short exit: ADX < 20 (range) OR Bull Power <= 0 (bearish momentum fading)
                if adx_val < 20 or bull <= 0:
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