#!/usr/bin/env python3
# Hypothesis: 6h Camarilla pivot breakout with 1d trend filter and volume confirmation.
# Uses daily Camarilla pivot levels (R3/S3 for reversal, R4/S4 for breakout).
# Long when price breaks above R4 with 1d uptrend; short when breaks below S4 with 1d downtrend.
# Reversals at R3/S3 with volume exhaustion. Low trade frequency (~15-25/year) to minimize fee drag.
# Camarilla levels work in ranging markets; breakouts capture trends in both bull and bear.

name = "6h_Camarilla_R3S4_Breakout_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close arrays."""
    typical = (high + low + close) / 3
    range_val = high - low
    R4 = close + range_val * 1.1 / 2
    R3 = close + range_val * 1.1 / 4
    S3 = close - range_val * 1.1 / 4
    S4 = close - range_val * 1.1 / 2
    return R4, R3, S3, S4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla and EMA50
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels on 1d
    R4_1d, R3_1d, S3_1d, S4_1d = calculate_camarilla(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values
    )
    
    # Camarilla needs no extra delay (based on completed daily bar)
    R4_1d_aligned = align_htf_to_ltf(prices, df_1d, R4_1d)
    R3_1d_aligned = align_htf_to_ltf(prices, df_1d, R3_1d)
    S3_1d_aligned = align_htf_to_ltf(prices, df_1d, S3_1d)
    S4_1d_aligned = align_htf_to_ltf(prices, df_1d, S4_1d)
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: current volume > 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma20
    
    # Volume exhaustion: current volume < 50% of 20-period average (for reversals)
    volume_exhausted = volume < (vol_ma20 * 0.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient data
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma20[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Breakout above R4 with uptrend and volume
            if (not np.isnan(R4_1d_aligned[i]) and 
                close[i] > R4_1d_aligned[i] and 
                close[i] > ema50_1d_aligned[i] and 
                volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below S4 with downtrend and volume
            elif (not np.isnan(S4_1d_aligned[i]) and 
                  close[i] < S4_1d_aligned[i] and 
                  close[i] < ema50_1d_aligned[i] and 
                  volume_ok[i]):
                signals[i] = -0.25
                position = -1
            # LONG reversal: bounce from S3 with volume exhaustion
            elif (not np.isnan(S3_1d_aligned[i]) and 
                  close[i] < S3_1d_aligned[i] * 1.005 and  # near S3
                  close[i] > ema50_1d_aligned[i] and       # above trend
                  volume_exhausted[i]):
                signals[i] = 0.20
                position = 1
            # SHORT reversal: rejection at R3 with volume exhaustion
            elif (not np.isnan(R3_1d_aligned[i]) and 
                  close[i] > R3_1d_aligned[i] * 0.995 and  # near R3
                  close[i] < ema50_1d_aligned[i] and       # below trend
                  volume_exhausted[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below R3 (failure) or EMA50 (trend change)
            if (not np.isnan(R3_1d_aligned[i]) and close[i] < R3_1d_aligned[i]) or \
               close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above S3 (failure) or EMA50 (trend change)
            if (not np.isnan(S3_1d_aligned[i]) and close[i] > S3_1d_aligned[i]) or \
               close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals