#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume confirmation
# Uses Camarilla pivot levels from weekly chart to identify major support/resistance.
# Enters long when price breaks above weekly R3 with volume confirmation and 1w EMA50 uptrend.
# Enters short when price breaks below weekly S3 with volume confirmation and 1w EMA50 downtrend.
# Weekly timeframe provides stronger trend filter than daily, reducing whipsaws in bear markets.
# Designed for 12-37 trades/year (~50-150 total over 4 years) to minimize fee drag.
# Weekly Camarilla levels offer more significant structure, reducing false breakouts.

name = "6h_Camarilla_R3S3_Breakout_1wEMA50_Trend_VolumeSpike"
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
    
    # Get 1w data for Camarilla calculation and EMA50 - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla levels
    camarilla_range = high_1w - low_1w
    r3_1w = close_1w + 1.1 * camarilla_range
    s3_1w = close_1w - 1.1 * camarilla_range
    
    # Align weekly Camarilla levels to 6h timeframe (wait for completed 1w bar)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    
    # Get 1w data for EMA50 trend filter - ONCE before loop
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA50 to 6h timeframe (wait for completed 1w bar)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate volume spike filter (20-period volume MA)
    vol_ma_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 2.0)  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(r3_1w_aligned[i]) or np.isnan(s3_1w_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above weekly R3 AND volume spike AND 1w EMA50 uptrend
            if (close[i] > r3_1w_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below weekly S3 AND volume spike AND 1w EMA50 downtrend
            elif (close[i] < s3_1w_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price re-enters weekly Camarilla range (between S3 and R3) OR trend reverses
            if (close[i] >= s3_1w_aligned[i] and close[i] <= r3_1w_aligned[i]) or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price re-enters weekly Camarilla range OR trend reverses
            if (close[i] >= s3_1w_aligned[i] and close[i] <= r3_1w_aligned[i]) or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals