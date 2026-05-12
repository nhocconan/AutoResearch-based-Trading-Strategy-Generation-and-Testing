#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike
Hypothesis: Camarilla pivot breakouts on 12h with daily trend filter and volume spike confirmation.
Enters long when price breaks above R1 with daily uptrend and volume > 1.5x average.
Enters short when price breaks below S1 with daily downtrend and volume > 1.5x average.
Uses daily trend to avoid counter-trend trades and volume spike to confirm institutional interest.
Designed for low trade frequency (12-37/year) to minimize fee drag in BTC/ETH markets.
"""

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close"""
    pivot = (high + low + close) / 3
    range_val = high - low
    r1 = pivot + (range_val * 1.1 / 4)
    r2 = pivot + (range_val * 1.1 / 2)
    r3 = pivot + (range_val * 1.1)
    s1 = pivot - (range_val * 1.1 / 4)
    s2 = pivot - (range_val * 1.1 / 2)
    s3 = pivot - (range_val * 1.1)
    return r1, r2, r3, s1, s2, s3

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    # Calculate Camarilla levels on 12h data (using previous bar's HLC)
    # Shift by 1 to avoid look-ahead (use previous bar to calculate levels for current bar)
    prev_high = np.concatenate([[np.nan], high[:-1]]) if len(high) > 1 else np.full_like(high, np.nan)
    prev_low = np.concatenate([[np.nan], low[:-1]]) if len(low) > 1 else np.full_like(low, np.nan)
    prev_close = np.concatenate([[np.nan], close[:-1]]) if len(close) > 1 else np.full_like(close, np.nan)
    
    r1, r2, r3, s1, s2, s3 = calculate_camarilla(prev_high, prev_low, prev_close)
    
    # Daily EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume average (20-period) for spike detection
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute daily trend alignment
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(1, n):  # Start from 1 to have previous bar data
        # Get aligned values for current 12h bar
        r1_val = r1[i]
        s1_val = s1[i]
        ema50_val = ema50_aligned[i]
        vol_avg_val = vol_avg_20[i]
        
        # Skip if any required data is NaN
        if (np.isnan(r1_val) or np.isnan(s1_val) or 
            np.isnan(ema50_val) or np.isnan(vol_avg_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R1 + daily uptrend + volume spike
            if (close[i] > r1_val and 
                close[i] > ema50_val and 
                volume[i] > vol_avg_val * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 + daily downtrend + volume spike
            elif (close[i] < s1_val and 
                  close[i] < ema50_val and 
                  volume[i] > vol_avg_val * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below R1 or trend turns down
            if (close[i] < r1_val or close[i] < ema50_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above S1 or trend turns up
            if (close[i] > s1_val or close[i] > ema50_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals