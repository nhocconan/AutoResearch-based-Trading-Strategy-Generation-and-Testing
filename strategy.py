#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R mean reversion with 12h ADX regime filter and volume confirmation.
- Long when Williams %R(14) < -80 (oversold) AND 12h ADX < 25 (range regime) AND volume > 1.2 * avg volume
- Short when Williams %R(14) > -20 (overbought) AND 12h ADX < 25 (range regime) AND volume > 1.2 * avg volume
- Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts) OR ADX > 30 (trend regime)
- Position size fixed at 0.25 to balance risk and avoid fee churn
- Uses 6h primary with 12h HTF to target 50-150 total trades over 4 years (12-37/year)
- Williams %R identifies exhaustion points in ranging markets; ADX filter avoids trending markets where mean reversion fails
- Volume confirmation ensures breakouts/breakdowns have participation, reducing false signals
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
    
    # Williams %R(14): (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # Average volume for confirmation (50-period SMA)
    avg_volume = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    # Get 12h data ONCE before loop for ADX regime filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h ADX(14)
    # True Range
    tr1 = pd.Series(df_12h['high'] - df_12h['low'])
    tr2 = pd.Series(np.abs(df_12h['high'] - np.roll(df_12h['close'], 1)))
    tr3 = pd.Series(np.abs(df_12h['low'] - np.roll(df_12h['close'], 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_12h = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    up_move = pd.Series(df_12h['high']).diff()
    down_move = pd.Series(df_12h['low']).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM and TR
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_12h
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_12h
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_12h = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 12h ADX to 6h timeframe
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Regime: ranging if ADX < 25, trending if ADX > 30 (with hysteresis)
    ranging_regime = adx_12h_aligned < 25
    trending_regime = adx_12h_aligned > 30
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(avg_volume[i]) or 
            np.isnan(adx_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.2 * 50-period average
        volume_confirmed = volume[i] > 1.2 * avg_volume[i]
        
        if position == 0:
            # Long: oversold + ranging regime + volume confirmation
            if williams_r[i] < -80 and ranging_regime[i] and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: overbought + ranging regime + volume confirmation
            elif williams_r[i] > -20 and ranging_regime[i] and volume_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses above -50 OR ADX > 30 (trend regime)
            if williams_r[i] > -50 or trending_regime[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses below -50 OR ADX > 30 (trend regime)
            if williams_r[i] < -50 or trending_regime[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_12hADX_Range_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0