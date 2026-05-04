#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike (>1.8x 20 EMA volume)
# Uses Camarilla pivot levels from prior completed 12h bar for structure (R3/S3 = fade zone, R4/S4 = breakout)
# 1d EMA34 filter ensures we trade in direction of longer-term trend (more robust for bear markets)
# Volume confirmation ensures breakout has sufficient participation (>1.8x average volume)
# Discrete sizing 0.25 balances risk and return while minimizing fee churn
# Target: 50-150 total trades over 4 years = 12-37/year for 12h timeframe
# Works in both bull (R4/S4 breakout continuation) and bear (R3/S3 fade + R4/S4 breakdown) markets
# Focus on BTC/ETH by requiring 1d trend alignment (avoids SOL-only bias, more robust across regimes)

name = "12h_Camarilla_R3S3_1dEMA34_VolumeSpike"
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
    
    # Get 1d data for Camarilla pivots and EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need enough data for EMA calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(34) trend filter from prior completed 1d bar
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_shifted = np.roll(ema_34_1d, 1)
    ema_34_1d_shifted[0] = np.nan
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d_shifted)
    
    # Calculate Camarilla pivot levels from prior completed 12h bar
    # Camarilla: P = (H+L+C)/3, R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), etc.
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r3 = close_1d + (range_1d * 1.1 / 4.0)
    r4 = close_1d + (range_1d * 1.1 / 2.0)
    s3 = close_1d - (range_1d * 1.1 / 4.0)
    s4 = close_1d - (range_1d * 1.1 / 2.0)
    
    # Shift by 1 to use only prior completed 1d bar (no look-ahead)
    r3_shifted = np.roll(r3, 1)
    r4_shifted = np.roll(r4, 1)
    s3_shifted = np.roll(s3, 1)
    s4_shifted = np.roll(s4, 1)
    r3_shifted[0] = np.nan
    r4_shifted[0] = np.nan
    s3_shifted[0] = np.nan
    s4_shifted[0] = np.nan
    
    # Align Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_shifted)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_shifted)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_shifted)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_shifted)
    
    # Volume confirmation: 20-period EMA of volume on 12h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R4 OR (price > R3 and price < R4) with volume spike AND price > 1d EMA34
            if (close[i] > r4_aligned[i] or (close[i] > r3_aligned[i] and close[i] < r4_aligned[i])) and \
               close[i] > ema_34_1d_aligned[i] and volume[i] > (1.8 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below S4 OR (price < S3 and price > S4) with volume spike AND price < 1d EMA34
            elif (close[i] < s4_aligned[i] or (close[i] < s3_aligned[i] and close[i] > s4_aligned[i])) and \
                 close[i] < ema_34_1d_aligned[i] and volume[i] > (1.8 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to S3 OR price crosses below 1d EMA34
            if close[i] < s3_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to R3 OR price crosses above 1d EMA34
            if close[i] > r3_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals