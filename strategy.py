#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_1wVol
Hypothesis: Camarilla pivot breakouts on 12h with daily trend filter and weekly volume confirmation.
Enters long when price breaks above R3 with daily uptrend and weekly volume above average.
Enters short when price breaks below S3 with daily downtrend and weekly volume above average.
Uses daily trend to avoid counter-trend trades and weekly volume to confirm institutional interest.
Designed for low trade frequency (12-37/year) to minimize fee drag in BTC/ETH markets.
"""

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_1wVol"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close"""
    pivot = (high + low + close) / 3
    range_val = high - low
    r3 = pivot + (range_val * 1.1 / 2)
    r4 = pivot + (range_val * 1.1)
    s3 = pivot - (range_val * 1.1 / 2)
    s4 = pivot - (range_val * 1.1)
    return r3, r4, s3, s4

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
    
    # Get weekly data for volume filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)

    # Calculate Camarilla levels on 12h data (using previous bar's HLC)
    # Shift by 1 to avoid look-ahead (use previous bar to calculate levels for current bar)
    prev_high = np.concatenate([[np.nan], high[:-1]]) if len(high) > 1 else np.full_like(high, np.nan)
    prev_low = np.concatenate([[np.nan], low[:-1]]) if len(low) > 1 else np.full_like(low, np.nan)
    prev_close = np.concatenate([[np.nan], close[:-1]]) if len(close) > 1 else np.full_like(close, np.nan)
    
    r3, r4, s3, s4 = calculate_camarilla(prev_high, prev_low, prev_close)
    
    # Daily EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Weekly average volume (20-period) for institutional interest filter
    vol_avg_20w = pd.Series(df_1w['volume']).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(1, n):  # Start from 1 to have previous bar data
        # Get aligned values for current 12h bar
        r3_val = r3[i]
        s3_val = s3[i]
        ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)[i]
        vol_avg_aligned = align_htf_to_ltf(prices, df_1w, vol_avg_20w)[i]
        vol_current = volume[i]
        
        # Skip if any required data is NaN
        if (np.isnan(r3_val) or np.isnan(s3_val) or 
            np.isnan(ema50_aligned) or np.isnan(vol_avg_aligned)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R3 + daily uptrend + weekly volume above average
            if (close[i] > r3_val and 
                close[i] > ema50_aligned and 
                vol_current > vol_avg_aligned):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 + daily downtrend + weekly volume above average
            elif (close[i] < s3_val and 
                  close[i] < ema50_aligned and 
                  vol_current > vol_avg_aligned):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below R3 or trend turns down
            if (close[i] < r3_val or close[i] < ema50_aligned):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above S3 or trend turns up
            if (close[i] > s3_val or close[i] > ema50_aligned):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals