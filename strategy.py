#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy combining 1-day Camarilla R3/S3 levels with 1-week trend filter and volume confirmation.
# Enters long when price breaks above R3 with weekly uptrend and volume spike, short when price breaks below S3 with weekly downtrend and volume spike.
# Exits on trend reversal or price crossing opposite level (R3<->S3). Uses daily levels with weekly trend to avoid look-ahead.
# Designed to work in both bull and bear markets by aligning with weekly trend. Target: 15-30 trades/year to minimize fee drag.

name = "12h_Camarilla_R3S3_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate EMA20 on 1w close for trend filter
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    # Camarilla R3, S3 levels: (H-L)*1.1/4
    camarilla_range = (high_1d - low_1d) * 1.1 / 4
    r3_level = close_1d_vals + camarilla_range
    s3_level = close_1d_vals - camarilla_range
    
    # Align Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_level)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_level)
    
    # Volume spike filter: current volume > 2.0 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Need enough data for EMA20 (1w) and volume MA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(ema20_1w_aligned[i]) or 
            np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema20_1w_val = ema20_1w_aligned[i]
        r3 = r3_aligned[i]
        s3 = s3_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: Close breaks above R3 + 1w uptrend + volume spike
            if close[i] > r3 and close[i] > ema20_1w_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: Close breaks below S3 + 1w downtrend + volume spike
            elif close[i] < s3 and close[i] < ema20_1w_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Close falls below S3 or 1w trend turns down
            if close[i] < s3 or close[i] < ema20_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Close rises above R3 or 1w trend turns up
            if close[i] > r3 or close[i] > ema20_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals