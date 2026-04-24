#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme Reversal with 12h ADX Trend Filter and Volume Spike.
- Primary timeframe: 6h for execution, HTF: 12h for ADX trend filter.
- Entry: Williams %R(14) crosses above -20 (long) or below -80 (short) on 6h close, with volume > 2.0x 20-period volume MA.
- Direction filter: only long when 12h ADX(14) > 25 AND +DI > -DI (uptrend), only short when ADX > 25 AND -DI > +DI (downtrend).
- Williams %R identifies overbought/oversold conditions; ADX filters for strong trends to avoid chop.
- Volume confirmation reduces false reversals.
- Exit: Williams %R returns to -50 (mean reversion) or trend filter reversal (ADX < 20 or DI crossover).
- Discrete signal size: 0.25 to balance return and drawdown control.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Works in bull via buying oversold pullbacks in uptrend, in bear via selling overbought bounces in downtrend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h ADX(14), +DI, -DI for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 12h indicators to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    di_plus_aligned = align_htf_to_ltf(prices, df_12h, di_plus)
    di_minus_aligned = align_htf_to_ltf(prices, df_12h, di_minus)
    
    # Calculate Williams %R(14) on 6h
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 20) + 14  # Need 12h ADX(30), volume MA(20), Williams %R(14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(di_plus_aligned[i]) or 
            np.isnan(di_minus_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R crosses above -20 (from below) with volume spike AND strong uptrend (ADX>25 and +DI>-DI)
            if (williams_r[i] > -20 and williams_r[i-1] <= -20 and volume_spike[i] and 
                adx_aligned[i] > 25 and di_plus_aligned[i] > di_minus_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -80 (from above) with volume spike AND strong downtrend (ADX>25 and -DI>+DI)
            elif (williams_r[i] < -80 and williams_r[i-1] >= -80 and volume_spike[i] and 
                  adx_aligned[i] > 25 and di_minus_aligned[i] > di_plus_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R returns to -50 (mean reversion) or trend weakens (ADX<20 or DI crossover)
            if (williams_r[i] < -50 or adx_aligned[i] < 20 or di_plus_aligned[i] < di_minus_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R returns to -50 (mean reversion) or trend weakens (ADX<20 or DI crossover)
            if (williams_r[i] > -50 or adx_aligned[i] < 20 or di_minus_aligned[i] < di_plus_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_12hADX_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0