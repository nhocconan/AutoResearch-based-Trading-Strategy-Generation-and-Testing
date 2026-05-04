#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume spike confirmation
# Uses 1h timeframe for precision entry timing, 4h EMA50 for trend direction, and daily Camarilla pivots for structure.
# Volume spike (2x 20-period EMA) confirms institutional participation. Designed for 15-30 trades/year on 1h to minimize fee drag.
# Works in bull markets via breakout continuations above R1 and in bear markets via breakdowns below S1.
# Session filter (08-20 UTC) reduces noise during low-liquidity hours.

name = "1h_Camarilla_R1S1_4hEMA50_VolumeSpike_SessionFilter"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    # Calculate 4h EMA50 from prior completed 4h bar
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_shifted = np.roll(ema50_4h, 1)
    ema50_4h_shifted[0] = np.nan
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h_shifted)
    
    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels from prior completed 1d bar
    R1_1d = close_1d + 1.125 * (high_1d - low_1d)
    S1_1d = close_1d - 1.125 * (high_1d - low_1d)
    R1_1d_shifted = np.roll(R1_1d, 1)
    S1_1d_shifted = np.roll(S1_1d, 1)
    R1_1d_shifted[0] = np.nan
    S1_1d_shifted[0] = np.nan
    R1_1d_aligned = align_htf_to_ltf(prices, df_1d, R1_1d_shifted)
    S1_1d_aligned = align_htf_to_ltf(prices, df_1d, S1_1d_shifted)
    
    # Volume confirmation: 20-period EMA of volume on 1h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(R1_1d_aligned[i]) or
            np.isnan(S1_1d_aligned[i]) or
            np.isnan(vol_ema_20[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R1 AND above 4h EMA50 AND volume spike
            if close[i] > R1_1d_aligned[i] and close[i] > ema50_4h_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: price breaks below S1 AND below 4h EMA50 AND volume spike
            elif close[i] < S1_1d_aligned[i] and close[i] < ema50_4h_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price closes below S1 OR below 4h EMA50
            if close[i] < S1_1d_aligned[i] or close[i] < ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price closes above R1 OR above 4h EMA50
            if close[i] > R1_1d_aligned[i] or close[i] > ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals