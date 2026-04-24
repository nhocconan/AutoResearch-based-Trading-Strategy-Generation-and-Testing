#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 1d ADX regime filter and volume confirmation.
- Bull Power = High - EMA13(close); Bear Power = EMA13(close) - Low
- Long when Bull Power > 0 AND Bear Power < 0 (bullish momentum) AND 1d ADX > 25 (trending) AND volume > 1.5x 20-period average volume
- Short when Bear Power > 0 AND Bull Power < 0 (bearish momentum) AND 1d ADX > 25 (trending) AND volume > 1.5x 20-period average volume
- Exit when momentum diverges (Bull Power <= 0 for long, Bear Power <= 0 for short) OR ADX < 20 (range) OR volume < average
- Uses 6h primary with 1d HTF to target 50-150 trades over 4 years (12-37/year)
- Elder Ray measures power of bulls/bears relative to EMA; ADX filters choppy markets; volume confirms conviction
- Works in bull trends (strong Bull Power) and bear trends (strong Bear Power) while avoiding whipsaws in ranging markets
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
    
    # Elder Ray Index components
    ema_len = 13
    ema_close = pd.Series(close).ewm(span=ema_len, adjust=False, min_periods=ema_len).mean().values
    bull_power = high - ema_close  # Bull Power: High - EMA(close)
    bear_power = ema_close - low   # Bear Power: EMA(close) - Low
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    # Get 1d data ONCE before loop for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period)
    dx_len = 14
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(np.maximum(tr1, tr2), tr3)
    tr_1d[0] = tr1[0]  # first value
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed TR, +DM, -DM
    atr_1d = pd.Series(tr_1d).ewm(span=dx_len, adjust=False, min_periods=dx_len).mean().values
    plus_di_1d = 100 * (pd.Series(plus_dm).ewm(span=dx_len, adjust=False, min_periods=dx_len).mean().values / atr_1d)
    minus_di_1d = 100 * (pd.Series(minus_dm).ewm(span=dx_len, adjust=False, min_periods=dx_len).mean().values / atr_1d)
    
    # DX and ADX
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d + 1e-10)
    adx_1d = pd.Series(dx_1d).ewm(span=dx_len, adjust=False, min_periods=dx_len).mean().values
    
    # Align 1d ADX to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Regime filters
    strong_trend = adx_1d_aligned > 25      # Trending market
    weak_trend = adx_1d_aligned < 20        # Ranging market (exit condition)
    
    # Momentum conditions
    bullish_momentum = (bull_power > 0) & (bear_power < 0)  # Bulls in control
    bearish_momentum = (bear_power > 0) & (bull_power < 0)  # Bears in control
    
    # Entry conditions
    long_entry = bullish_momentum & strong_trend & volume_confirm
    short_entry = bearish_momentum & strong_trend & volume_confirm
    
    # Exit conditions
    long_exit = (~bullish_momentum) | weak_trend | (~volume_confirm)
    short_exit = (~bearish_momentum) | weak_trend | (~volume_confirm)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(ema_len, 20, dx_len) + 5
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(volume_confirm[i]) or np.isnan(adx_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Look for entry
            if long_entry[i]:
                signals[i] = 0.25
                position = 1
            elif short_entry[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position - check exit
            if long_exit[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position - check exit
            if short_exit[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_1dADX_Regime_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0