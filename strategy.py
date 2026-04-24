#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d ADX regime filter + volume confirmation.
- Bull Power = High - EMA13(close); Bear Power = EMA13(close) - Low
- Long when Bull Power > 0 AND Bear Power < 0 AND 1d ADX > 25 (strong trend) AND volume > 1.5x 20-period average
- Short when Bear Power > 0 AND Bull Power < 0 AND 1d ADX > 25 AND volume > 1.5x 20-period average
- Exit when Elder Power signals weaken (Bull Power <= 0 for long, Bear Power <= 0 for short) OR volume drops below average
- Uses 6h primary with 1d HTF for ADX regime to avoid whipsaws in ranging markets
- Elder Ray measures trend strength via price relative to EMA; ADX confirms trend persistence
- Volume filter ensures participation; discrete sizing (0.25) minimizes fee churn
- Target: 50-150 total trades over 4 years (12-37/year) with strong trend capture in both bull/bear regimes
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
    
    # Elder Ray components: EMA13 of close
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull Power: High - EMA13
    bear_power = ema13 - low   # Bear Power: EMA13 - Low
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    # Get 1d data ONCE before loop for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX(14)
    # True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # Directional Movement
    up_move = df_1d['high'] - df_1d['high'].shift(1)
    down_move = df_1d['low'].shift(1) - df_1d['low']
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    # Smoothed TR, +DM, -DM
    tr_rma = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_dm_rma = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_rma = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    # Directional Indicators
    plus_di = 100 * plus_dm_rma / tr_rma
    minus_di = 100 * minus_dm_rma / tr_rma
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Regime: strong trend when ADX > 25
    strong_trend = adx_aligned > 25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # Fixed size to minimize fee churn
    
    # Start from index where all indicators are ready
    start_idx = max(13, 20, 14) + 10  # Elder Ray EMA13, volume MA20, ADX14 with smoothing
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power < 0 AND strong trend AND volume filter
            if (bull_power[i] > 0 and bear_power[i] < 0 and 
                strong_trend[i] and volume_filter[i]):
                signals[i] = position_size
                position = 1
            # Short: Bear Power > 0 AND Bull Power < 0 AND strong trend AND volume filter
            elif (bear_power[i] > 0 and bull_power[i] < 0 and 
                  strong_trend[i] and volume_filter[i]):
                signals[i] = -position_size
                position = -1
        elif position == 1:
            # Long exit: Bull Power <= 0 OR volume drops below average
            if bull_power[i] <= 0 or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = position_size
        elif position == -1:
            # Short exit: Bear Power <= 0 OR volume drops below average
            if bear_power[i] <= 0 or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_ElderRay_1dADX_Regime_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0