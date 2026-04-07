#!/usr/bin/env python3
"""
12h_camarilla_pivot_1d_volume_v2
Hypothesis: Daily Camarilla pivot reversal with volume confirmation on 12h chart.
Long when price rejects S3/S4 with volume spike; short when rejected at R3/R4.
Uses only daily pivots + volume to minimize trades (<20/year) and avoid overtrading.
Works in bull/bear via mean-reversion at institutional levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_volume_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC
    prev_high = np.roll(df_1d['high'].values, 1)
    prev_low = np.roll(df_1d['low'].values, 1)
    prev_close = np.roll(df_1d['close'].values, 1)
    prev_high[0] = df_1d['high'].values[0]
    prev_low[0] = df_1d['low'].values[0]
    prev_close[0] = df_1d['close'].values[0]
    
    # Camarilla levels
    range_1d = prev_high - prev_low
    camarilla_S3 = prev_close - (range_1d * 1.1 / 6)
    camarilla_S4 = prev_close - (range_1d * 1.1 / 4)
    camarilla_R3 = prev_close + (range_1d * 1.1 / 6)
    camarilla_R4 = prev_close + (range_1d * 1.1 / 4)
    
    # Align to 12h
    S3 = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    S4 = align_htf_to_ltf(prices, df_1d, camarilla_S4)
    R3 = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    R4 = align_htf_to_ltf(prices, df_1d, camarilla_R4)
    
    # Volume spike: 2x 24-period average (2 days worth)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(24, n):
        if (np.isnan(S3[i]) or np.isnan(S4[i]) or np.isnan(R3[i]) or np.isnan(R4[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            continue
        
        if position == 1:
            # Exit long: price crosses S4 or loses volume momentum
            if close[i] < S4[i] or not vol_spike[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses R4 or loses volume momentum
            if close[i] > R3[i] or not vol_spike[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:
            # Enter long: rejection of S3/S4 with volume spike
            if ((close[i] <= S3[i] * 1.005) or (close[i] <= S4[i] * 1.005)) and vol_spike[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: rejection of R3/R4 with volume spike
            elif ((close[i] >= R3[i] * 0.995) or (close[i] >= R4[i] * 0.995)) and vol_spike[i]:
                position = -1
                signals[i] = -0.25
    
    return signals