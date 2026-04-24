#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Power (Bull/Bear) with 1d ADX regime filter and volume confirmation.
- Primary timeframe: 6h for execution, HTF: 1d for Elder Ray and ADX.
- ADX > 25 indicates trending market: go long when Bull Power > 0, short when Bear Power < 0.
- ADX < 20 indicates ranging market: fade extremes (long when Bull Power crosses above 0 from below,
  short when Bear Power crosses below 0 from above).
- Volume confirmation: current 6h volume > 1.5 * 20-period volume MA to avoid false signals.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Works in both bull and bear markets via regime adaptation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Elder Ray and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Elder Ray Power (Bull Power = High - EMA13, Bear Power = Low - EMA13)
    ema13 = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = df_1d['high'].values - ema13
    bear_power = df_1d['low'].values - ema13
    
    # Calculate ADX (14-period) on 1d
    # True Range
    tr1 = pd.Series(df_1d['high']).diff().abs()
    tr2 = (pd.Series(df_1d['high']) - pd.Series(df_1d['low'].shift())).abs()
    tr3 = (pd.Series(df_1d['low']) - pd.Series(df_1d['close'].shift())).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    up_move = pd.Series(df_1d['high']).diff()
    down_move = -pd.Series(df_1d['low']).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / (atr + 1e-10)
    minus_di = 100 * minus_dm_smooth / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align HTF indicators to 6h
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA (on 6h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 20)  # Need enough 1d bars for indicators and 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_val = adx_aligned[i]
        curr_bull = bull_power_aligned[i]
        curr_bear = bear_power_aligned[i]
        prev_bull = bull_power_aligned[i-1]
        prev_bear = bear_power_aligned[i-1]
        
        if position == 0:
            # Check for entry signals
            if volume_spike[i]:
                if adx_val > 25:  # Trending regime: follow Elder Ray power
                    # Bullish: Bull Power > 0
                    if curr_bull > 0:
                        signals[i] = 0.25
                        position = 1
                    # Bearish: Bear Power < 0
                    elif curr_bear < 0:
                        signals[i] = -0.25
                        position = -1
                else:  # Ranging regime (ADX < 20): fade extremes via zero crosses
                    # Long when Bull Power crosses above 0 from below
                    if prev_bull <= 0 and curr_bull > 0:
                        signals[i] = 0.25
                        position = 1
                    # Short when Bear Power crosses below 0 from above
                    elif prev_bear >= 0 and curr_bear < 0:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: Bull Power turns negative OR ADX drops to ranging
            if curr_bull <= 0 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power turns positive OR ADX drops to ranging
            if curr_bear >= 0 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_Power_1dADXRegime_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0