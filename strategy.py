#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1w EMA(50) trend filter and volume spike (>1.8x 24 EMA volume)
# Uses Camarilla pivot levels from prior completed 1d bar for structure (R3/S3 = strong support/resistance)
# 1w EMA(50) filter ensures we only trade in the direction of the higher timeframe trend
# Volume confirmation ensures breakout has sufficient participation (>1.8x average volume)
# Discrete sizing 0.25 balances risk and return while minimizing fee churn
# Target: 80-160 total trades over 4 years = 20-40/year for 12h timeframe
# Works in both bull (breakouts continuation) and bear (breakdowns continuation) markets
# Focus on BTC/ETH by requiring 1w trend alignment (avoids SOL-only bias)

name = "12h_Camarilla_R3S3_1wEMA50_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need enough data for EMA calculation
        return np.zeros(n)
    
    # Calculate 1w EMA(50) trend filter from prior completed 1w bar
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_shifted = np.roll(ema_50_1w, 1)
    ema_50_1w_shifted[0] = np.nan
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w_shifted)
    
    # Get 1d data for Camarilla pivot levels (prior completed 1d bar)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:  # Need at least 1 day for pivot calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: R3, S3 from prior completed 1d bar
    def camarilla_levels(high, low, close):
        # Typical price for Camarilla calculation
        typical = (high + low + close) / 3.0
        # Camarilla width
        width = high - low
        # R3 = close + width * 1.1/4
        # S3 = close - width * 1.1/4
        r3 = close + width * 1.1 / 4.0
        s3 = close - width * 1.1 / 4.0
        return r3, s3
    
    r3, s3 = camarilla_levels(high_1d, low_1d, close_1d)
    
    # Shift by 1 to use only prior completed 1d bar (no look-ahead)
    r3_shifted = np.roll(r3, 1)
    s3_shifted = np.roll(s3, 1)
    r3_shifted[0] = np.nan
    s3_shifted[0] = np.nan
    
    # Align Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_shifted)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_shifted)
    
    # Volume confirmation: 24-period EMA of volume on 12h timeframe (2*12h = 1d)
    vol_ema_24 = pd.Series(volume).ewm(span=24, adjust=False, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ema_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R3 + price > 1w EMA50 + volume spike
            if close[i] > r3_aligned[i] and close[i] > ema_50_1w_aligned[i] and volume[i] > (1.8 * vol_ema_24[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below S3 + price < 1w EMA50 + volume spike
            elif close[i] < s3_aligned[i] and close[i] < ema_50_1w_aligned[i] and volume[i] > (1.8 * vol_ema_24[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to S3 OR price crosses below 1w EMA50
            if close[i] < s3_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to R3 OR price crosses above 1w EMA50
            if close[i] > r3_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals