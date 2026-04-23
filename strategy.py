#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d ADX regime filter and volume confirmation.
- Bull Power = High - EMA13(close); Bear Power = EMA13(close) - Low
- Long: Bull Power > 0 AND Bear Power < 0 AND ADX(1d) > 25 AND volume > 1.5x 20-period avg
- Short: Bear Power > 0 AND Bull Power < 0 AND ADX(1d) > 25 AND volume > 1.5x 20-period avg
- Exit: Opposite Elder Ray signal OR ADX(1d) < 20 (regime change to ranging)
- Uses 1d HTF for ADX to filter trending markets only (avoids whipsaws in ranges)
- Elder Ray captures trend strength via power of bulls/bears relative to EMA
- Designed for moderate trade frequency (12-37/year) on 6h timeframe
- Works in bull (strong bull power) and bear (strong bear power) markets
- Volume confirmation ensures participation
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
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d EMA13 for Elder Ray (HTF = 1d)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 1d ADX for regime filter (HTF = 1d)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.abs(high_1d[0] - low_1d[0])  # First period
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = -np.diff(low_1d, prepend=low_1d[0])
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing = EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_1d = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    bull_power = high - ema_13_1d
    bear_power = ema_13_1d - low
    
    # Align Elder Ray components to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 13, 14)  # Need 20 for vol MA, 13 for EMA, 14 for ADX
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(bull_power_aligned[i]) or
            np.isnan(bear_power_aligned[i]) or
            np.isnan(adx_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Elder Ray signals
        bull_strong = bull_power_aligned[i] > 0 and bear_power_aligned[i] < 0
        bear_strong = bear_power_aligned[i] > 0 and bull_power_aligned[i] < 0
        
        # ADX regime filter: trending market only
        trending = adx_1d_aligned[i] > 25
        ranging = adx_1d_aligned[i] < 20
        
        if position == 0:
            # Long: Bull power positive, bear power negative, trending, volume confirmation
            if bull_strong and trending and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Bear power positive, bull power negative, trending, volume confirmation
            elif bear_strong and trending and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bear power becomes positive OR ADX drops to ranging
            if bear_power_aligned[i] > 0 or ranging:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bull power becomes positive OR ADX drops to ranging
            if bull_power_aligned[i] > 0 or ranging:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_ADXRegime_VolumeConfirm"
timeframe = "6h"
leverage = 1.0