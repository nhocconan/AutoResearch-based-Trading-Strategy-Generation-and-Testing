#!/usr/bin/env python3
# 4H_CAMARILLA_R3_S3_BREAKOUT_1D_VOLUME_SPIKE
# Hypothesis: Camarilla R3/S3 levels from daily chart act as strong support/resistance.
# Breakouts above R3 or below S3 with volume spike confirmation capture momentum moves.
# Volume filter reduces false breakouts and improves win rate. Works in bull and bear markets.
# Target: 20-40 trades/year on 4h timeframe.

name = "4H_CAMARILLA_R3_S3_BREAKOUT_1D_VOLUME_SPIKE"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
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
    
    # R3 = C + (H-L)*1.25/2, S3 = C - (H-L)*1.25/2
    r3 = close_1d + (high_1d - low_1d) * 1.25 / 2
    s3 = close_1d - (high_1d - low_1d) * 1.25 / 2
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Align to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 1  # Need at least one day of data
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(volume_spike_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 with volume spike
            if (close[i] > r3_aligned[i] and volume_spike_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 with volume spike
            elif (close[i] < s3_aligned[i] and volume_spike_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls below S3
            if close[i] < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises above R3
            if close[i] > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals