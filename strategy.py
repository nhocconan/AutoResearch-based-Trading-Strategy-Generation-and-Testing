#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d timeframe for direction and 12h for entry timing.
# Uses 1d Camarilla pivot levels (R3/S3) with 12h close confirmation and volume spike.
# Designed to work in both bull and bear markets by trading reversals at extreme pivot levels.
# Target: 12-37 trades per year to minimize fee drag and improve generalization.
name = "12h_Camarilla_R3S3_Reversal_Volume"
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
    
    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate previous day's Camarilla levels (using prior day's OHLC)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate pivot point
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    # Camarilla levels: R3/S3 are the most significant for reversals
    r3 = pivot + (range_hl * 1.1 / 2.0)  # R3 = C + 1.1*(H-L)/2
    s3 = pivot - (range_hl * 1.1 / 2.0)  # S3 = C - 1.1*(H-L)/2
    
    # Align 1d Camarilla levels to 12h timeframe (wait for 1d bar to close)
    r3_12h = align_htf_to_ltf(prices, df_1d, r3)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume filter: spike above 2.0x 4-period average (2 days of 12h bars)
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 4  # Wait for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r3_12h[i]) or np.isnan(s3_12h[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 2.0 * vol_ma[i]  # Volume confirmation
        
        # Pre-compute hour for session filter (UTC 0-24)
        hour = pd.DatetimeIndex(prices['open_time']).hour[i]
        # Trade during active hours (8 AM - 8 PM UTC) to avoid low liquidity
        in_session = (8 <= hour <= 20)
        
        if position == 0:
            # Long reversal: price touches/bounces off S3 with volume
            if (low[i] <= s3_12h[i] and close[i] > s3_12h[i] and 
                vol_ok and in_session):
                signals[i] = 0.25
                position = 1
            # Short reversal: price touches/rejects at R3 with volume
            elif (high[i] >= r3_12h[i] and close[i] < r3_12h[i] and 
                  vol_ok and in_session):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price reaches R3 or shows weakness
            if close[i] >= r3_12h[i] or low[i] < s3_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price reaches S3 or shows strength
            if close[i] <= s3_12h[i] or high[i] > r3_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals