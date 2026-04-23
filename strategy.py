#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R3/S3 breakout with 1d ATR filter and volume spike confirmation.
- Camarilla pivot levels (R3, S3) from 1d timeframe act as strong intraday support/resistance
- Long: Close breaks above R3 + volume > 2.0x 20-period average + 1d ATR(14) > 0.015 * price
- Short: Close breaks below S3 + volume > 2.0x 20-period average + 1d ATR(14) > 0.015 * price
- Exit: Opposite Camarilla break (S3 for long, R3 for short) or ATR filter fails
- Uses Camarilla levels for structure, volume for conviction, ATR for volatility regime filter
- Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe
- Discrete position sizing: ±0.25 to minimize fee churn
- Camarilla breakouts work in both bull (breakouts in uptrend) and bear (breakdowns in downtrend) markets
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
    
    # Volume confirmation: > 2.0x 20-period average (tight to avoid overtrading)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d Camarilla pivot levels (R3, S3)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formulas: R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    camarilla_r3_1d = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_s3_1d = close_1d - (high_1d - low_1d) * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe (no extra delay needed for pivot points)
    camarilla_r3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    
    # Calculate 1d ATR(14) for volatility regime filter
    tr1 = pd.Series(high_1d - low_1d).values
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1))).values
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1))).values
    tr2[0] = tr1[0]  # First bar: no previous close
    tr3[0] = tr1[0]
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14)  # Need 20 for volume MA, 14 for ATR
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(camarilla_r3_1d_aligned[i]) or
            np.isnan(camarilla_s3_1d_aligned[i]) or
            np.isnan(atr_14_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # ATR filter: volatility regime (ATR > 1.5% of price)
        atr_filter = atr_14_1d_aligned[i] > 0.015 * close[i]
        
        if position == 0:
            # Long: Close breaks above R3 + volume confirmation + ATR filter
            if (close[i] > camarilla_r3_1d_aligned[i] and 
                volume_confirm and 
                atr_filter):
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below S3 + volume confirmation + ATR filter
            elif (close[i] < camarilla_s3_1d_aligned[i] and 
                  volume_confirm and 
                  atr_filter):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close breaks below S3 OR ATR filter fails
            if (close[i] < camarilla_s3_1d_aligned[i]) or not atr_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close breaks above R3 OR ATR filter fails
            if (close[i] > camarilla_r3_1d_aligned[i]) or not atr_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_1dATR_VolumeSpike"
timeframe = "12h"
leverage = 1.0