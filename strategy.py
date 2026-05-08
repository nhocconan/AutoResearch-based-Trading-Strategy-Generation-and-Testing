#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Camarilla pivot breakout with weekly trend filter and volume confirmation
# We go long when price breaks above R3 with weekly EMA(34) uptrend and volume spike.
# We go short when price breaks below S3 with weekly EMA(34) downtrend and volume spike.
# Uses 4h timeframe targeting 20-50 trades/year, avoiding excessive frequency.
# Weekly trend filter ensures we trade with the higher timeframe momentum.
# Volume spike confirms institutional participation in the breakout.
# Weekly timeframe provides more stable trend filter than daily, reducing false signals.

name = "4h_Camarilla_R3S3_WeeklyTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data once for Camarilla pivots and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly EMA(34) for trend filter
    weekly_close = df_1w['close'].values
    ema34_1w = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate Camarilla pivot levels from weekly data
    # R3 = close + 1.1*(high - low)/2
    # S3 = close - 1.1*(high - low)/2
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close_vals = df_1w['close'].values
    
    camarilla_r3 = weekly_close_vals + 1.1 * (weekly_high - weekly_low) / 2
    camarilla_s3 = weekly_close_vals - 1.1 * (weekly_high - weekly_low) / 2
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    
    # Volume spike: current volume > 2.0 * 20-period average on 4h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_1w_val = ema34_1w_aligned[i]
        r3_level = camarilla_r3_aligned[i]
        s3_level = camarilla_s3_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price breaks above R3 + weekly uptrend + volume spike
            if (not np.isnan(r3_level) and close[i] > r3_level and 
                close[i] > ema34_1w_val and vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S3 + weekly downtrend + volume spike
            elif (not np.isnan(s3_level) and close[i] < s3_level and 
                  close[i] < ema34_1w_val and vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S3 OR weekly trend turns down
            if (not np.isnan(s3_level) and close[i] < s3_level) or close[i] < ema34_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R3 OR weekly trend turns up
            if (not np.isnan(r3_level) and close[i] > r3_level) or close[i] > ema34_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals