#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R3S3 breakout with 1d ATR volatility filter and volume spike confirmation.
- Long: Close breaks above R3 + volume > 2.0x 20-period avg + ATR(14) > 0.5x ATR(50) (vol expansion)
- Short: Close breaks below S3 + volume > 2.0x 20-period avg + ATR(14) > 0.5x ATR(50) (vol expansion)
- Exit: Close retouches the pivot point (PP) from opposite side or ATR contraction < 0.3x ATR(50)
- Uses Camarilla levels for institutional price reaction, volume for conviction, ATR regime filter
- Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe
- Discrete position sizing: ±0.25 to minimize fee churn
- Works in bull (breakouts continue) and bear (breakdowns continue) via volatility expansion
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
    
    # Volume confirmation: > 2.0x 20-period average (strict to avoid overtrading)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR regime filter: ATR(14) > 0.5x ATR(50) for expansion, < 0.3x for contraction
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    atr_expansion = atr_14 > (0.5 * atr_50)
    atr_contraction = atr_14 < (0.3 * atr_50)
    
    # Calculate 1d Camarilla levels (R3, S3, PP)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla calculations: PP = (H+L+C)/3, R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    r3_1d = close_1d + ((high_1d - low_1d) * 1.1 / 2.0)
    s3_1d = close_1d - ((high_1d - low_1d) * 1.1 / 2.0)
    
    # Align HTF levels to LTF
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 14)  # Need 50 for ATR50, 20 for volume MA, 14 for ATR14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(atr_14[i]) or 
            np.isnan(atr_50[i]) or
            np.isnan(pp_aligned[i]) or
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # ATR regime: expansion for entry, contraction for exit
        vol_expanding = atr_expansion[i]
        vol_contracting = atr_contraction[i]
        
        if position == 0:
            # Long: Close breaks above R3 + volume confirmation + volatility expanding
            if (close[i] > r3_aligned[i] and 
                volume_confirm and 
                vol_expanding):
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below S3 + volume confirmation + volatility expanding
            elif (close[i] < s3_aligned[i] and 
                  volume_confirm and 
                  vol_expanding):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close retouches PP from below OR volatility contraction
            if (close[i] < pp_aligned[i] and 
                (i == start_idx or close[i-1] >= pp_aligned[i-1])) or vol_contracting:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close retouches PP from above OR volatility contraction
            if (close[i] > pp_aligned[i] and 
                (i == start_idx or close[i-1] <= pp_aligned[i-1])) or vol_contracting:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_1dATR_VolumeSpike"
timeframe = "12h"
leverage = 1.0