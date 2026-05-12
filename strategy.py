#!/usr/bin/env python3
# 4H_CAMARILLA_R1_S1_BREAKOUT_1D_TREND_VOLUME_SPIKE
# Hypothesis: Combines daily Camarilla pivot breakouts with 4h trend filter and volume spike confirmation.
# R1/S1 from daily chart provide tighter levels than R3/S3, reducing false breakouts.
# Trend filter uses 4h EMA50 to align with intermediate trend.
# Volume spike requires current volume > 1.5x 20-period average to confirm breakout strength.
# Works in bull markets (breakouts continuation) and bear markets (reversals at extremes).
# Target: 20-40 trades/year on 4h timeframe.

name = "4H_CAMARILLA_R1_S1_BREAKOUT_1D_TREND_VOLUME_SPIKE"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # R1 = C + (H-L)*1.0833/2, S1 = C - (H-L)*1.0833/2
    r1 = close_1d + (high_1d - low_1d) * 1.0833 / 2
    s1 = close_1d - (high_1d - low_1d) * 1.0833 / 2
    
    # 4h EMA50 for trend filter
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    # Align to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 1  # Need at least one day of data
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(ema50_aligned[i]) or np.isnan(volume_spike[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R1 in uptrend with volume spike
            if (close[i] > r1_aligned[i] and 
                close[i] > ema50_aligned[i] and
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 in downtrend with volume spike
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema50_aligned[i] and
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls below S1 or trend reversal
            if (close[i] < s1_aligned[i] or 
                close[i] <= ema50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises above R1 or trend reversal
            if (close[i] > r1_aligned[i] or 
                close[i] >= ema50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals