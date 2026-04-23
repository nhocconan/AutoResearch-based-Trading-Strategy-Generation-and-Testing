#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R3/S3 breakout with volume spike and ATR expansion filter.
- Long: Close breaks above Camarilla R3 + volume > 2.0x 20-period avg + ATR(14) > ATR(50)
- Short: Close breaks below Camarilla S3 + volume > 2.0x 20-period avg + ATR(14) > ATR(50)
- Exit: Close crosses Camarilla H6/L6 levels (extreme mean reversion)
- Uses Camarilla pivot levels from daily HTF for structure, volume spike for momentum confirmation,
  and ATR expansion filter to ensure volatility regime (avoids choppy markets)
- Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe
- Discrete position sizing: ±0.25 to minimize fee churn while maintaining exposure
- Works in bull markets (breakouts with volume/volatility expansion) and bear markets (mean reversion at extreme pivots)
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
    
    # Volume confirmation: > 2.0x 20-period average (stricter volume filter to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR calculation for volatility regime filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Calculate Camarilla pivot levels from 1d HTF data
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    camarilla_range = high_1d - low_1d
    camarilla_h6 = close_1d + camarilla_range * 1.1
    camarilla_l6 = close_1d - camarilla_range * 1.1
    camarilla_h3 = close_1d + camarilla_range * 1.1 / 6
    camarilla_l3 = close_1d - camarilla_range * 1.1 / 6
    camarilla_r3 = camarilla_h3  # R3 = H3
    camarilla_s3 = camarilla_l3  # S3 = L3
    
    # Align HTF levels to LTF (12h) - ensures no look-ahead, uses completed 1d bars
    camarilla_h6_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h6)
    camarilla_l6_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l6)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 50 for ATR50, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(atr_14[i]) or
            np.isnan(atr_50[i]) or
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(camarilla_h6_aligned[i]) or
            np.isnan(camarilla_l6_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility regime filter: ATR(14) > ATR(50) (expanding volatility)
        vol_regime = atr_14[i] > atr_50[i]
        
        # Volume confirmation (> 2.0x average - stricter to reduce trade frequency)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Close breaks above Camarilla R3 + volatility expansion + volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                vol_regime and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below Camarilla S3 + volatility expansion + volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  vol_regime and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close crosses below Camarilla L6 (extreme mean reversion)
            if close[i] < camarilla_l6_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close crosses above Camarilla H6 (extreme mean reversion)
            if close[i] > camarilla_h6_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_VolumeSpike_ATRExpansion"
timeframe = "12h"
leverage = 1.0