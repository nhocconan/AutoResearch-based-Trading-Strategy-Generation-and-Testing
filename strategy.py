#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d ATR regime filter and volume confirmation.
- Primary timeframe: 12h, HTF: 1d for ATR regime filter
- Long: Close breaks above Donchian upper (20-period high) + ATR(14) < ATR(50) (low volatility regime) + volume > 1.5x 20-period avg
- Short: Close breaks below Donchian lower (20-period low) + ATR(14) < ATR(50) (low volatility regime) + volume > 1.5x 20-period avg
- Exit: Close reverts to 20-period midpoint (mean reversion)
- Uses Donchian breakouts in low volatility regimes to capture explosive moves with controlled entries
- Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe
- Discrete position sizing: ±0.25 to balance return and risk
- BTC/ETH focus: requires ATR regime filter to avoid false breakouts in choppy markets
- Works in bull markets (breakouts with momentum) and bear markets (breakdowns with momentum)
- Uses mtf_data helper for proper HTF alignment without look-ahead
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation: > 1.5x 20-period average (volume spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Donchian channels (20-period) from 12h data
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_ma + low_ma) / 2.0
    
    # Calculate 1d ATR for regime filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First period
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) and ATR(50)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Low volatility regime: ATR(14) < ATR(50) (volatility contraction)
    low_vol_regime = atr_14 < atr_50
    
    # Align to 12h timeframe (values from previous 1d bar)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_50_aligned = align_htf_to_ltf(prices, df_1d, atr_50)
    low_vol_aligned = align_htf_to_ltf(prices, df_1d, low_vol_regime.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # Need 20 for Donchian, 50 for ATR(50)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(high_ma[i]) or 
            np.isnan(low_ma[i]) or 
            np.isnan(donchian_mid[i]) or 
            np.isnan(atr_14_aligned[i]) or 
            np.isnan(atr_50_aligned[i]) or 
            np.isnan(low_vol_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation (> 1.5x average)
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        # Low volatility regime filter
        low_vol = low_vol_aligned[i] > 0.5
        
        if position == 0:
            # Long: Close breaks above Donchian upper + low vol regime + volume spike
            if (close[i] > high_ma[i] and 
                low_vol and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below Donchian lower + low vol regime + volume spike
            elif (close[i] < low_ma[i] and 
                  low_vol and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close reverts to Donchian midpoint (mean reversion)
            if close[i] <= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close reverts to Donchian midpoint (mean reversion)
            if close[i] >= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dATRRegime_VolumeSpike"
timeframe = "12h"
leverage = 1.0