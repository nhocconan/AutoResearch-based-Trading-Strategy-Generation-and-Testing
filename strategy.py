#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike (>2.0x 20 EMA volume)
# Uses Camarilla levels from prior completed 1d bar for structure (R3/S3 = proven breakout levels with good follow-through)
# 12h EMA50 filter ensures we only trade in the direction of the higher timeframe trend, reducing whipsaw
# Volume confirmation ensures breakout has strong participation
# Discrete sizing 0.25 balances risk and return while minimizing fee churn
# Target: 80-150 total trades over 4 years = 20-38/year for 4h timeframe
# This strategy focuses on BTC/ETH by using HTF trend and volume filters that work across market regimes

name = "4h_Camarilla_R3S3_12hEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need enough data for EMA50 calculation
        return np.zeros(n)
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough data for Camarilla calculation
        return np.zeros(n)
    
    # Calculate 12h EMA50 trend filter from prior completed 12h bar
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_shifted = np.roll(ema_50_12h, 1)  # Use prior completed 12h bar
    ema_50_12h_shifted[0] = np.nan
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h_shifted)
    
    # Volume confirmation: 20-period EMA of volume on 4h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate Camarilla levels (R3, S3) from prior completed 1d bar
    # Camarilla formula: R3 = close + 1.1*(high-low)/4, S3 = close - 1.1*(high-low)/4
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_val = df_1d['close'].values
    
    camarilla_range = high_1d - low_1d
    r3 = close_1d_val + (1.1 * camarilla_range / 4)
    s3 = close_1d_val - (1.1 * camarilla_range / 4)
    
    # Shift by 1 to use only prior completed 1d bar
    r3_shifted = np.roll(r3, 1)
    s3_shifted = np.roll(s3, 1)
    r3_shifted[0] = np.nan
    s3_shifted[0] = np.nan
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_shifted)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_shifted)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ema_20[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R3 + price > 12h EMA50 (uptrend) + volume spike
            if close[i] > r3_aligned[i] and close[i] > ema_50_12h_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below S3 + price < 12h EMA50 (downtrend) + volume spike
            elif close[i] < s3_aligned[i] and close[i] < ema_50_12h_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to midpoint of Camarilla levels OR price crosses below 12h EMA50
            midpoint = (r3_aligned[i] + s3_aligned[i]) / 2
            if not np.isnan(midpoint) and (close[i] < midpoint or close[i] < ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to midpoint of Camarilla levels OR price crosses above 12h EMA50
            midpoint = (r3_aligned[i] + s3_aligned[i]) / 2
            if not np.isnan(midpoint) and (close[i] > midpoint or close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals