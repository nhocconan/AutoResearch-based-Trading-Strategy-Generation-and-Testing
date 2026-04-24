#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d ADX25 regime filter and volume spike confirmation.
- Primary timeframe: 6h, HTF: 1d for ADX regime and EMA13 trend.
- Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 (using 13-period EMA).
- Regime filter: Only trade when 1d ADX > 25 (trending market) to avoid whipsaws in ranging markets.
- Volume confirmation: Current 6h volume > 1.5 * 20-period 6h volume MA.
- Entry: Long when Bull Power > 0 AND Bear Power < previous Bear Power (bullish momentum) in uptrend regime.
          Short when Bear Power < 0 AND Bull Power < previous Bull Power (bearish momentum) in downtrend regime.
- Exit: Reverse signal or when power decays (Bull Power < 0 for long, Bear Power > 0 for short).
- Discrete signal size: 0.25 to balance profit potential and drawdown control.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Works in both bull/bear: ADX filter ensures we only trade strong trends, Elder Ray captures momentum within trend.
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
    
    # Calculate 1d EMA13 for Elder Ray (using 13 as standard)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Calculate 1d ADX for regime filter (standard 14-period)
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d_arr, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d_arr, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smooth TR, DM+, DM- (using Wilder's smoothing = EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where((di_plus + di_minus) == 0, 0, dx)  # Avoid division by zero
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Elder Ray components
    bull_power = high - ema_13_1d_aligned
    bear_power = low - ema_13_1d_aligned
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    # Regime filter: trending market (ADX > 25)
    trending_regime = adx_aligned > 25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 14, 20)  # Need ADX, EMA13, volume MA
    
    # Initialize previous power values
    prev_bull_power = bull_power[0] if len(bull_power) > 0 else 0
    prev_bear_power = bear_power[0] if len(bear_power) > 0 else 0
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_13_1d_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            # Update previous values for next iteration
            prev_bull_power = bull_power[i] if not np.isnan(bull_power[i]) else prev_bull_power
            prev_bear_power = bear_power[i] if not np.isnan(bear_power[i]) else prev_bear_power
            continue
        
        if position == 0:
            # Long: Bull Power > 0 (price above EMA13) AND Bull Power rising (momentum) in uptrend regime
            if bull_power[i] > 0 and bull_power[i] > prev_bull_power and trending_regime[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (price below EMA13) AND Bear Power falling (momentum) in downtrend regime
            elif bear_power[i] < 0 and bear_power[i] < prev_bear_power and trending_regime[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bull Power <= 0 (momentum lost) or reverse signal
            if bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power >= 0 (momentum lost) or reverse signal
            if bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        
        # Update previous power values for next iteration
        prev_bull_power = bull_power[i]
        prev_bear_power = bear_power[i]
    
    return signals

name = "6h_ElderRay_1dADX25_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0