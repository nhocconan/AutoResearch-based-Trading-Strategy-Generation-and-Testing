#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d ADX25 regime filter and volume confirmation.
- Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 (using 1d EMA13 for smoothing).
- Regime filter: Only trade when 1d ADX > 25 (trending market) to avoid whipsaws in ranging periods.
- Entry: Long when Bull Power > 0 and rising (current > previous) AND volume spike.
         Short when Bear Power < 0 and falling (current < previous) AND volume spike.
- Volume confirmation: Current 6h volume > 1.8 * 20-period 6h volume MA (moderate to balance signal quality and frequency).
- Discrete signal size: 0.25 to manage drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Works in both bull/bear: ADX regime ensures we only trade strong trends, Elder Ray captures momentum with trend alignment.
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
    
    # Calculate 1d EMA13 for Elder Ray smoothing
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Calculate 1d ADX for regime filter
    if len(df_1d) < 14:  # ADX needs at least 14 periods
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align length
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / np.where(atr == 0, np.nan, atr)
    di_minus = 100 * dm_minus_smooth / np.where(atr == 0, np.nan, atr)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / np.where((di_plus + di_minus) == 0, np.nan, (di_plus + di_minus))
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Elder Ray components
    bull_power = high - ema_13_1d_aligned
    bear_power = low - ema_13_1d_aligned
    
    # Volume confirmation: current volume > 1.8 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 14)  # Need ADX and sufficient volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_13_1d_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(volume_spike[i]) or i == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 AND rising (current > previous) AND ADX > 25 AND volume spike
            if (bull_power[i] > 0 and i > 0 and bull_power[i] > bull_power[i-1] and 
                adx_aligned[i] > 25 and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND falling (current < previous) AND ADX > 25 AND volume spike
            elif (bear_power[i] < 0 and i > 0 and bear_power[i] < bear_power[i-1] and 
                  adx_aligned[i] > 25 and volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bull Power <= 0 (momentum lost) OR reverse signal
            if bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power >= 0 (momentum lost) OR reverse signal
            if bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_1dADX25_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0