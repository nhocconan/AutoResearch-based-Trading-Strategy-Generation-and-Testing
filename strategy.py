#!/usr/bin/env python3
# 12h_Camarilla_R3S3_VolumeSpike_Breakout
# Hypothesis: On 12h timeframe, trade breakouts from daily Camarilla R3/S3 levels with volume spike confirmation.
# Uses 2x volume spike above 20-period MA for confirmation. Includes volatility filter (ATR-based) to avoid choppy markets.
# Targets 12-30 trades per year. Works in bull/bear via price action at institutional levels.

name = "12h_Camarilla_R3S3_VolumeSpike_Breakout"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price for pivot calculation
    typical_price_1d = (high_1d + low_1d + close_1d) / 3
    
    # Pivot point and ranges
    pivot_1d = typical_price_1d
    range_1d = high_1d - low_1d
    
    # Camarilla levels: R3, S3
    s3_1d = close_1d - (range_1d * 1.1 / 4)
    r3_1d = close_1d + (range_1d * 1.1 / 4)
    
    # Align daily levels to 12h timeframe
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    
    # Volume average for spike detection (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Volatility filter: ATR(14) to avoid choppy markets
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(s3_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: require ATR > 0 to avoid dead markets
        if atr[i] <= 0:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout above R3 with volume confirmation
            if (close[i] > r3_aligned[i] * 1.005 and 
                volume[i] > 2.0 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short breakdown below S3 with volume
            elif (close[i] < s3_aligned[i] * 0.995 and 
                  volume[i] > 2.0 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: breakdown below S3
            if close[i] < s3_aligned[i] * 0.995:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: breakout above R3
            if close[i] > r3_aligned[i] * 1.005:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals