#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with daily trend filter and volume spike
# Uses daily Camarilla levels for entry, 1d EMA(50) for trend, and volume > 2x 20-period average.
# Designed for low trade frequency (20-50/year) with clear entry/exit rules.
# Works in both bull and bear markets by following trend direction from higher timeframe.
# Target: 40-120 total trades over 4 years (10-30/year)

name = "4h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels (based on previous day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R3, R2, R1, PP, S1, S2, S3
    # Using previous day's OHLC
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]  # fill first value
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Calculate pivot point and ranges
    pp = (prev_high + prev_low + prev_close) / 3.0
    rang = prev_high - prev_low
    
    # Camarilla levels
    r3 = pp + (rang * 1.1 / 4.0)
    r1 = pp + (rang * 1.1 / 12.0)
    s1 = pp - (rang * 1.1 / 12.0)
    s3 = pp - (rang * 1.1 / 4.0)
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Daily EMA(50) for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike: current volume > 2.0 * 20-period average (using 4h data)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        r3_val = r3_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        s3_val = s3_aligned[i]
        ema50_val = ema50_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price breaks above R3 + daily uptrend + volume spike
            if (close[i] > r3_val and 
                close[i] > ema50_val and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S3 + daily downtrend + volume spike
            elif (close[i] < s3_val and 
                  close[i] < ema50_val and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S1 OR daily trend turns down
            if (close[i] < s1_val or close[i] < ema50_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R1 OR daily trend turns up
            if (close[i] > r1_val or close[i] > ema50_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals