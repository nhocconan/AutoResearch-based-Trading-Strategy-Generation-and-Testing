#!/usr/bin/env python3
# 4h_Camarilla_Pivot_Reversal_1dTrend_Volume
# Hypothesis: Price rejection at Camarilla pivot levels (S3/R3) on 4h with 1d trend filter and volume confirmation.
# In ranging markets, price reverses at S3/R3; in trending markets, breaks through these levels continue the trend.
# The 1d trend (via EMA50) filters for direction, and volume confirms conviction.
# Designed to work in both bull and bear markets by adapting to regime via price action at key levels.

name = "4h_Camarilla_Pivot_Reversal_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 1d EMA50 for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Calculate Camarilla levels from previous day's range ===
    # We need prior day's high/low/close to calculate today's Camarilla
    # Since we're on 4h timeframe, we'll use the prior day's data
    # We'll calculate Camarilla for each day and align to 4h
    
    # Get daily OHLC for Camarilla calculation
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    # R4 = close + (high - low) * 1.500
    # R3 = close + (high - low) * 1.250
    # R2 = close + (high - low) * 1.166
    # R1 = close + (high - low) * 1.083
    # S1 = close - (high - low) * 1.083
    # S2 = close - (high - low) * 1.166
    # S3 = close - (high - low) * 1.250
    # S4 = close - (high - low) * 1.500
    
    # We use previous day's data to avoid look-ahead
    prev_daily_high = np.roll(daily_high, 1)
    prev_daily_low = np.roll(daily_low, 1)
    prev_daily_close = np.roll(daily_close, 1)
    # First day will have NaN due to roll, which is correct (no prior day)
    
    # Calculate Camarilla levels using previous day's data
    R3 = prev_daily_close + (prev_daily_high - prev_daily_low) * 1.250
    S3 = prev_daily_close - (prev_daily_high - prev_daily_low) * 1.250
    
    # Align Camarilla levels to 4h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # === Volume confirmation (20-period average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter: above average
        vol_ok = volume[i] > vol_ma_20[i]
        
        # Determine trend from 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # LONG: Price rejects S3 (bounces off support) in uptrend OR breaks above R3 in any trend with volume
            if ((close[i] > S3_aligned[i] and close[i-1] <= S3_aligned[i-1]) or  # bounce off S3
                (close[i] > R3_aligned[i] and close[i-1] <= R3_aligned[i-1])) and vol_ok:  # break above R3
                signals[i] = 0.25
                position = 1
            # SHORT: Price rejects R3 (bounces off resistance) in downtrend OR breaks below S3 in any trend with volume
            elif ((close[i] < R3_aligned[i] and close[i-1] >= R3_aligned[i-1]) or  # bounce off R3
                  (close[i] < S3_aligned[i] and close[i-1] >= S3_aligned[i-1])) and vol_ok:  # break below S3
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price fails to hold above S3 or breaks below R3 in uptrend context
            if close[i] < S3_aligned[i] or (downtrend and close[i] < R3_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price fails to hold below R3 or breaks above S3 in downtrend context
            if close[i] > R3_aligned[i] or (uptrend and close[i] > S3_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals