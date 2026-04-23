#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R3/S3 breakout with 1d ATR regime filter and volume confirmation.
- Primary timeframe: 12h, HTF: 1d for ATR-based regime detection
- Long: Close breaks above R3 + ATR(14) > ATR(50) (high volatility regime) + volume > 1.5x 20-period avg
- Short: Close breaks below S3 + ATR(14) > ATR(50) (high volatility regime) + volume > 1.5x 20-period avg
- Exit: Close reverts to pivot point (PP) of Camarilla levels
- Uses wider Camarilla breakouts (R3/S3) to capture stronger moves while reducing trade frequency
- ATR regime filter ensures trades only occur in high volatility environments (trending markets)
- Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe
- Discrete position sizing: ±0.25 to balance return and risk
- BTC/ETH focus: requires volatility expansion to avoid choppy/range-bound losing periods
- Works in bull markets (breakouts with volatility expansion) and bear markets (breakdowns with volatility expansion)
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
    
    # Calculate ATR for regime filter (1d timeframe)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First bar: no previous close
    tr2[0] = np.abs(high_1d[0] - close_1d[0])  # First bar: use same bar close
    tr3[0] = np.abs(low_1d[0] - close_1d[0])   # First bar: use same bar close
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) and ATR(50) for regime detection
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Regime: high volatility when ATR(14) > ATR(50)
    high_vol_regime = atr_14 > atr_50
    
    # Calculate Camarilla levels from previous day using 1d data
    range_1d = high_1d - low_1d
    r3 = close_1d + 1.1 * range_1d / 4.0   # R3 = close + 1.1*(high-low)/4
    s3 = close_1d - 1.1 * range_1d / 4.0   # S3 = close - 1.1*(high-low)/4
    pp = (high_1d + low_1d + close_1d) / 3.0  # Pivot point
    
    # Align to 12h timeframe (values from previous 1d bar)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_50_aligned = align_htf_to_ltf(prices, df_1d, atr_50)
    high_vol_regime_aligned = align_htf_to_ltf(prices, df_1d, high_vol_regime.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # Need 20 for volume MA, 50 for ATR(50)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(pp_aligned[i]) or 
            np.isnan(atr_14_aligned[i]) or 
            np.isnan(atr_50_aligned[i]) or 
            np.isnan(high_vol_regime_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation (> 1.5x average)
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        # High volatility regime filter
        in_high_vol = high_vol_regime_aligned[i] > 0.5
        
        if position == 0:
            # Long: Close breaks above R3 + high volatility regime + volume spike
            if (close[i] > r3_aligned[i] and 
                in_high_vol and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below S3 + high volatility regime + volume spike
            elif (close[i] < s3_aligned[i] and 
                  in_high_vol and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close reverts to pivot point (PP)
            if close[i] <= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close reverts to pivot point (PP)
            if close[i] >= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_1dATR_Regime_VolumeSpike"
timeframe = "12h"
leverage = 1.0