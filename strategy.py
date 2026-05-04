#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 12h EMA34 trend filter and volume spike (>1.8x 20 EMA volume)
# Uses Camarilla pivot levels from prior completed 12h bar for structure (R3/S3 = fade zone, R4/S4 = breakout)
# 12h EMA34 filter ensures we trade in direction of intermediate trend (more responsive than 1d, less noisy than 6h)
# Volume confirmation ensures breakout has sufficient participation (>1.8x average volume)
# Discrete sizing 0.25 balances risk and return while minimizing fee churn
# Target: 50-150 total trades over 4 years = 12-37/year for 6h timeframe
# Works in both bull (R4/S4 breakout continuation) and bear (R3/S3 fade + R4/S4 breakdown) markets
# Focus on BTC/ETH by requiring 12h trend alignment (avoids SOL-only bias, more robust across regimes)

name = "6h_Camarilla_R3S3_12hEMA34_VolumeSpike"
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
    
    # Get 12h data for Camarilla pivots and EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:  # Need enough data for EMA calculation
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA(34) trend filter from prior completed 12h bar
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_shifted = np.roll(ema_34_12h, 1)
    ema_34_12h_shifted[0] = np.nan
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h_shifted)
    
    # Calculate Camarilla pivot levels from prior completed 12h bar
    # Camarilla: P = (H+L+C)/3, R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), etc.
    pivot = (high_12h + low_12h + close_12h) / 3.0
    range_12h = high_12h - low_12h
    r3 = close_12h + (range_12h * 1.1 / 4.0)
    r4 = close_12h + (range_12h * 1.1 / 2.0)
    s3 = close_12h - (range_12h * 1.1 / 4.0)
    s4 = close_12h - (range_12h * 1.1 / 2.0)
    
    # Shift by 1 to use only prior completed 12h bar (no look-ahead)
    r3_shifted = np.roll(r3, 1)
    r4_shifted = np.roll(r4, 1)
    s3_shifted = np.roll(s3, 1)
    s4_shifted = np.roll(s4, 1)
    r3_shifted[0] = np.nan
    r4_shifted[0] = np.nan
    s3_shifted[0] = np.nan
    s4_shifted[0] = np.nan
    
    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3_shifted)
    r4_aligned = align_htf_to_ltf(prices, df_12h, r4_shifted)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3_shifted)
    s4_aligned = align_htf_to_ltf(prices, df_12h, s4_shifted)
    
    # Volume confirmation: 20-period EMA of volume on 6h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R4 OR (price > R3 and price < R4) with volume spike AND price > 12h EMA34
            if (close[i] > r4_aligned[i] or (close[i] > r3_aligned[i] and close[i] < r4_aligned[i])) and \
               close[i] > ema_34_12h_aligned[i] and volume[i] > (1.8 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below S4 OR (price < S3 and price > S4) with volume spike AND price < 12h EMA34
            elif (close[i] < s4_aligned[i] or (close[i] < s3_aligned[i] and close[i] > s4_aligned[i])) and \
                 close[i] < ema_34_12h_aligned[i] and volume[i] > (1.8 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to S3 OR price crosses below 12h EMA34
            if close[i] < s3_aligned[i] or close[i] < ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to R3 OR price crosses above 12h EMA34
            if close[i] > r3_aligned[i] or close[i] > ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals