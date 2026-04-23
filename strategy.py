#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d ATR regime filter and volume confirmation.
- Primary timeframe: 12h, HTF: 1d for ATR regime filter
- Long: Close breaks above 20-period high + ATR(14)/ATR(50) < 0.8 (low volatility regime) + volume > 1.5x 20-period avg
- Short: Close breaks below 20-period low + ATR(14)/ATR(50) < 0.8 + volume > 1.5x 20-period avg
- Exit: Close reverts to 10-period moving average (mean reversion in low volatility)
- Uses Donchian breakouts in low volatility regimes to capture expansion moves with reduced false breakouts
- Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe
- Discrete position sizing: ±0.25 to minimize fee churn
- Works in bull markets (breakouts with trend continuation) and bear markets (breakdowns with trend continuation)
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
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 12h Donchian channels (20-period)
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 10-period MA for exit (mean reversion target)
    ma_10 = pd.Series(close).rolling(window=10, min_periods=10).mean().values
    
    # Calculate 1d ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # First value has no previous close
    tr2[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) and ATR(50)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # ATR ratio: ATR(14)/ATR(50) < 0.8 indicates low volatility regime
    atr_ratio = np.where(atr_50 != 0, atr_14 / atr_50, 1.0)
    
    # Align 1d indicators to 12h timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50, 10)  # Need 20 for Donchian/volume, 50 for ATR(50), 10 for MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(high_ma[i]) or 
            np.isnan(low_ma[i]) or 
            np.isnan(ma_10[i]) or 
            np.isnan(atr_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation (> 1.5x average)
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        # Low volatility regime filter
        low_vol_regime = atr_ratio_aligned[i] < 0.8
        
        if position == 0:
            # Long: Close breaks above 20-period high + low vol regime + volume spike
            if (close[i] > high_ma[i] and 
                low_vol_regime and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below 20-period low + low vol regime + volume spike
            elif (close[i] < low_ma[i] and 
                  low_vol_regime and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close reverts to 10-period MA (mean reversion)
            if close[i] <= ma_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close reverts to 10-period MA (mean reversion)
            if close[i] >= ma_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dATRRegime_VolumeSpike"
timeframe = "12h"
leverage = 1.0