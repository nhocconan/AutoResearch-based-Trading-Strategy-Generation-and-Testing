#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Mean Reversion with 12h ADX Regime Filter and Volume Confirmation.
- Long when Williams %R(14) < -80 (oversold) AND 12h ADX < 25 (low trend = range) AND volume > 1.5x ATR(14) * close
- Short when Williams %R(14) > -20 (overbought) AND 12h ADX < 25 (low trend = range) AND volume > 1.5x ATR(14) * close
- Exit when Williams %R crosses back above -50 (for longs) or below -50 (for shorts) OR ADX > 30 (trend emergence)
- Uses 6h primary timeframe with 12h HTF for regime filter to target 50-150 trades over 4 years (12-37/year)
- Williams %R identifies extreme price levels in ranging markets
- 12h ADX < 25 ensures we only mean-revert in low-volatility, range-bound conditions (avoids trending whipsaws)
- Volume confirmation filters low-momentum false signals
- Designed for BTC/ETH where ranging regimes are common in bear markets (2022, 2025+) and bull market consolidations
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
    
    # Calculate Williams %R(14) using lookback window (no look-ahead)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Get 12h data ONCE before loop for ADX regime filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h ADX(14)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr_12h = np.nanmax(np.column_stack([tr1, tr2, tr3]), axis=1)
    
    # Directional Movement
    up_move = np.diff(high_12h, prepend=np.nan)
    down_move = -np.diff(low_12h, prepend=np.nan)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_dm_14 = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_14 = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_14 / tr_14
    minus_di = 100 * minus_dm_14 / tr_14
    
    # ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx_12h = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 12h ADX to 6h timeframe
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Calculate ATR(14) for dynamic volume threshold (6h)
    tr1_6h = high - low
    tr2_6h = np.abs(high - np.roll(close, 1))
    tr3_6h = np.abs(low - np.roll(close, 1))
    tr2_6h[0] = np.nan
    tr3_6h[0] = np.nan
    tr_6h = np.nanmax(np.column_stack([tr1_6h, tr2_6h, tr3_6h]), axis=1)
    atr_6h = pd.Series(tr_6h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Dynamic volume threshold: volume > 1.5 * ATR * close (volatility-adjusted)
    vol_threshold = 1.5 * atr_6h * close
    volume_confirm = volume > vol_threshold
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 30, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(adx_12h_aligned[i]) or 
            np.isnan(atr_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: oversold + low ADX (range) + volume confirmation
            if williams_r[i] < -80 and adx_12h_aligned[i] < 25 and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: overbought + low ADX (range) + volume confirmation
            elif williams_r[i] > -20 and adx_12h_aligned[i] < 25 and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses above -50 OR ADX > 30 (trend emergence)
            if williams_r[i] > -50 or adx_12h_aligned[i] > 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses below -50 OR ADX > 30 (trend emergence)
            if williams_r[i] < -50 or adx_12h_aligned[i] > 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_12hADXRegime_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0