#!/usr/bin/env python3
# 12h_Camarilla_Pivot_1dTrend_Volume
# Hypothesis: Use 1d Camarilla pivot levels (S3/R3) for breakout entries in direction of 1d EMA(34) trend, with volume confirmation.
# In bull markets: buy when price breaks above R3 and price > EMA(34); in bear markets: sell when price breaks below S3 and price < EMA(34).
# Volume filter ensures breakouts have conviction. Exit when price returns to S4/R4 or trend reverses.
# Target: 20-30 trades/year to stay under fee drag limits.

name = "12h_Camarilla_Pivot_1dTrend_Volume"
timeframe = "12h"
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
    
    # 1d Camarilla pivot levels (based on previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    R4 = close_1d + (high_1d - low_1d) * 1.1 / 2
    R3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    R2 = close_1d + (high_1d - low_1d) * 1.1 / 6
    R1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    S1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    S2 = close_1d - (high_1d - low_1d) * 1.1 / 6
    S3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    S4 = close_1d - (high_1d - low_1d) * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe
    R3_1d_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_1d_aligned = align_htf_to_ltf(prices, df_1d, S3)
    R4_1d_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_1d_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # 1d trend filter: EMA(34)
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(R3_1d_aligned[i]) or np.isnan(S3_1d_aligned[i]) or
            np.isnan(R4_1d_aligned[i]) or np.isnan(S4_1d_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3, 1d trend up, volume confirmation
            if close[i] > R3_1d_aligned[i] and close[i] > ema_34_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3, 1d trend down, volume confirmation
            elif close[i] < S3_1d_aligned[i] and close[i] < ema_34_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price returns to S4 or trend changes
            if close[i] <= S4_1d_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price returns to R4 or trend changes
            if close[i] >= R4_1d_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals