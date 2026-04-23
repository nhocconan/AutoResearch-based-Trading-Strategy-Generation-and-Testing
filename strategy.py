#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 1d ATR volatility filter and volume confirmation.
- Uses 4h Camarilla pivot levels (R3, S3) from previous 1d OHLC for institutional breakout signals
- Long: price > R3 + volume > 1.5x 20-period avg + ATR(14) > 0.8x 50-period avg ATR (sufficient volatility)
- Short: price < S3 + volume > 1.5x 20-period avg + ATR(14) > 0.8x 50-period avg ATR
- Exit: price reverts to Camarilla pivot point (PP)
- ATR filter avoids low-volatility false breakouts that cause whipsaws
- Target: 20-40 trades/year (75-150 total over 4 years) to minimize fee drag on 4h timeframe
- Works in bull/bear regimes as Camarilla levels adapt to recent volatility
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
    
    # Volume confirmation: > 1.5x 20-period average (spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR calculation for volatility filter
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with original indices
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # ATR filter: current ATR > 0.8x 50-period average ATR (avoid low volatility)
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    vol_filter = atr > 0.8 * atr_ma
    
    # Load 1d data ONCE before loop for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from previous 1d OHLC
    # Camarilla formulas:
    # PP = (high + low + close) / 3
    # R3 = PP + (high - low) * 1.1 / 2
    # S3 = PP - (high - low) * 1.1 / 2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pp = (high_1d + low_1d + close_1d) / 3.0
    r3 = pp + (high_1d - low_1d) * 1.1 / 2.0
    s3 = pp - (high_1d - low_1d) * 1.1 / 2.0
    
    # Align Camarilla levels to 4h timeframe (wait for 1d bar close)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50, 14)  # Need 20 for volume MA, 50 for ATR MA, 14 for ATR
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(atr_ma[i]) or 
            np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(pp_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation (> 1.5x average) + volatility filter
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        sufficient_vol = vol_filter[i]
        
        if position == 0:
            # Long breakout: price > R3 + volume spike + sufficient volatility
            if volume_spike and sufficient_vol:
                if close[i] > r3_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            # Short breakdown: price < S3 + volume spike + sufficient volatility
            elif volume_spike and sufficient_vol:
                if close[i] < s3_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price reverts to pivot point (PP)
            if close[i] <= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to pivot point (PP)
            if close[i] >= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_ATR_VolumeSpike"
timeframe = "4h"
leverage = 1.0