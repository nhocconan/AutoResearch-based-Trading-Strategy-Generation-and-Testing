#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d volume spike and 1w trend filter
# Uses Camarilla pivot levels from daily timeframe for precise entry/exit levels,
# combined with 1d volume spike confirmation and 1w EMA50 trend filter to avoid counter-trend trades.
# Designed for 12-35 trades/year (~50-140 total over 4 years) to minimize fee drag.
# Camarilla R3/S3 provides mean-reversion fade logic, while breaks of R4/S4 indicate continuation.
# Volume spike confirms institutional participation. 1w EMA50 ensures alignment with weekly trend.

name = "6h_Camarilla_R3S3_Breakout_1dVolumeSpike_1wTrendFilter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and volume spike - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d EMA20 for volume average
    vol_ema20_1d = pd.Series(volume_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate Camarilla pivot levels for 1d: based on previous day's OHLC
    # R4 = close + 1.5 * (high - low), R3 = close + 1.1 * (high - low)
    # S3 = close - 1.1 * (high - low), S4 = close - 1.5 * (high - low)
    hl_range_1d = high_1d - low_1d
    camarilla_r4_1d = close_1d + 1.5 * hl_range_1d
    camarilla_r3_1d = close_1d + 1.1 * hl_range_1d
    camarilla_s3_1d = close_1d - 1.1 * hl_range_1d
    camarilla_s4_1d = close_1d - 1.5 * hl_range_1d
    
    # Align Camarilla levels to 6h timeframe (wait for completed 1d bar)
    camarilla_r4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4_1d)
    camarilla_r3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    camarilla_s4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4_1d)
    
    # Volume spike condition: current 1d volume > 2.0 * 20-period EMA of volume
    volume_spike_1d = volume_1d > (2.0 * vol_ema20_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Get 1w data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate 1w EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3_1d_aligned[i]) or np.isnan(camarilla_s3_1d_aligned[i]) or
            np.isnan(camarilla_r4_1d_aligned[i]) or np.isnan(camarilla_s4_1d_aligned[i]) or
            np.isnan(volume_spike_1d_aligned[i]) or np.isnan(ema50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R4 with volume spike AND above weekly EMA50
            if (close[i] > camarilla_r4_1d_aligned[i] and 
                volume_spike_1d_aligned[i] and 
                close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S4 with volume spike AND below weekly EMA50
            elif (close[i] < camarilla_s4_1d_aligned[i] and 
                  volume_spike_1d_aligned[i] and 
                  close[i] < ema50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price re-enters Camarilla R3-S3 range OR weekly trend turns bearish
            if (close[i] <= camarilla_r3_1d_aligned[i] and close[i] >= camarilla_s3_1d_aligned[i]) or \
               (close[i] < ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price re-enters Camarilla R3-S3 range OR weekly trend turns bullish
            if (close[i] <= camarilla_r3_1d_aligned[i] and close[i] >= camarilla_s3_1d_aligned[i]) or \
               (close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals