#!/usr/bin/env python3
"""
Hypothesis: 12h TRIX(12) signal line crossover with 1d volume spike and ADX(14) trend filter.
- Long: TRIX crosses above signal line + volume > 2.0x 20-period average + ADX > 25 (trending market)
- Short: TRIX crosses below signal line + volume > 2.0x 20-period average + ADX > 25 (trending market)
- Exit: Opposite TRIX crossover or volume drops below average
- Uses TRIX momentum oscillator for trend changes, volume confirmation for breakout strength,
  and ADX to ensure we only trade in trending regimes (avoiding choppy markets)
- Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe
- Discrete position sizing: ±0.25 to balance return and minimize fee churn
- Works in bull markets (momentum continuation) and bear markets (strong trend moves)
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
    
    # Volume confirmation: > 2.0x 20-period average (strict volume filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ADX calculation for trend regime filter
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smooth TR and DM
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # Calculate DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # Calculate DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # TRIX calculation: triple EMA of ROC
    roc = pd.Series(close).pct_change(periods=1).values  # 1-period ROC
    roc[0] = 0
    
    # Triple EMA of ROC
    ema1 = pd.Series(roc).ewm(span=12, min_periods=12, adjust=False).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, min_periods=12, adjust=False).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, min_periods=12, adjust=False).mean().values
    trix = ema3 * 100  # Scale for readability
    
    # TRIX signal line (EMA of TRIX)
    trix_signal = pd.Series(trix).ewm(span=9, min_periods=9, adjust=False).mean().values
    
    # Calculate 1d HTF volume for confirmation
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 20, 14)  # Need 30 for TRIX stability, 20 for volume MA, 14 for ADX
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(adx[i]) or
            np.isnan(trix[i]) or
            np.isnan(trix_signal[i]) or
            np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average - strict filter)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Trend regime filter: ADX > 25 (strong trend)
        trend_regime = adx[i] > 25
        
        # TRIX crossover signals
        trix_cross_up = trix[i] > trix_signal[i] and trix[i-1] <= trix_signal[i-1]
        trix_cross_down = trix[i] < trix_signal[i] and trix[i-1] >= trix_signal[i-1]
        
        if position == 0:
            # Long: TRIX bullish crossover + volume spike + trending market
            if (trix_cross_up and 
                volume_confirm and 
                trend_regime):
                signals[i] = 0.25
                position = 1
            # Short: TRIX bearish crossover + volume spike + trending market
            elif (trix_cross_down and 
                  volume_confirm and 
                  trend_regime):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX bearish crossover OR volume drops below average
            if trix_cross_down or volume[i] < vol_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX bullish crossover OR volume drops below average
            if trix_cross_up or volume[i] < vol_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_TRIX_VolumeSpike_ADXTrend"
timeframe = "12h"
leverage = 1.0