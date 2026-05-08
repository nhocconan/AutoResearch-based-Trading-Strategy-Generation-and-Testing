#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1w EMA20 trend filter and volume spike confirmation
# Designed for 20-40 trades/year with proper risk control via trend failure
# Long: price breaks above Camarilla R3 + price > 1w EMA20 + volume spike
# Short: price breaks below Camarilla S3 + price < 1w EMA20 + volume spike
# Exit: trend failure (price crosses 1w EMA20) or opposite breakout
# Volume filter: current 1d volume > 1.8x 20-day average to avoid false breakouts
# Camarilla levels provide strong intraday support/resistance, EMA20 on weekly filters trend, volume confirms breakout strength

name = "4h_Camarilla_R3S3_1wEMA20_VolumeFilter"
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
    
    # Get 1d data for Camarilla calculation and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w EMA20 for trend filter
    ema20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 20-day average volume for volume filter
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Camarilla levels (R3, S3) from 1d data
    # Camarilla: R3 = close + 1.1*(high-low)/1.1, S3 = close - 1.1*(high-low)/1.1
    # Simplified: R3 = close + (high-low), S3 = close - (high-low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    camarilla_r3 = close_1d + (high_1d - low_1d)
    camarilla_s3 = close_1d - (high_1d - low_1d)
    
    # Align 1w and 1d indicators to 4h timeframe
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup period
    
    for i in range(start_idx, n):
        if (np.isnan(ema20_1w_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ma_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current 1d volume > 1.8x 20-day average
        # Find the most recent completed 1d bar
        idx_1d = len(df_1d) - 1
        while idx_1d >= 0 and df_1d.iloc[idx_1d]['open_time'] > prices.iloc[i]['open_time']:
            idx_1d -= 1
        vol_filter = False
        if idx_1d >= 0:
            vol_1d_current = df_1d.iloc[idx_1d]['volume']
            vol_filter = vol_1d_current > 1.8 * vol_ma_20_aligned[i]
        
        if position == 0:
            # Look for breakout with trend and volume confirmation
            # Long: price breaks above Camarilla R3 + uptrend + volume spike
            if close[i] > camarilla_r3_aligned[i] and ema20_1w_aligned[i] > 0:
                if vol_filter:
                    signals[i] = 0.25
                    position = 1
            # Short: price breaks below Camarilla S3 + downtrend + volume spike
            elif close[i] < camarilla_s3_aligned[i] and ema20_1w_aligned[i] < 0:
                if vol_filter:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: trend failure (price crosses below EMA20) or opposite breakout
            if ema20_1w_aligned[i] <= 0 or close[i] < camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: trend failure (price crosses above EMA20) or opposite breakout
            if ema20_1w_aligned[i] >= 0 or close[i] > camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals