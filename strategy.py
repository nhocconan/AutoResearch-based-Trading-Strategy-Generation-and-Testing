#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA(34) trend filter and volume spike (>1.8x 20 EMA volume)
# Uses Camarilla pivot levels from prior completed 4h bar for structure (breakout = price > R3 or < S3)
# 12h EMA(34) filter ensures we only trade in the direction of the higher timeframe trend
# Volume confirmation ensures breakout has sufficient participation (>1.8x average volume)
# Discrete sizing 0.25 balances risk and return while minimizing fee churn
# Target: 75-150 total trades over 4 years = 19-38/year for 4h timeframe
# Works in both bull (breakouts continuation) and bear (breakdowns continuation) markets
# Focus on BTC/ETH by requiring 12h trend alignment (more robust than 1d alone)

name = "4h_Camarilla_R3S3_12hEMA34_VolumeSpike"
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
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:  # Need enough data for EMA calculation
        return np.zeros(n)
    
    # Calculate 12h EMA(34) trend filter from prior completed 12h bar
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_shifted = np.roll(ema_34_12h, 1)
    ema_34_12h_shifted[0] = np.nan
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h_shifted)
    
    # Get 4h data for Camarilla pivot levels (prior completed 4h bar)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:  # Need at least one completed 4h bar for pivot calculation
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels for each 4h bar: based on prior completed bar
    # R3 = close + (high - low) * 1.1/4
    # S3 = close - (high - low) * 1.1/4
    camarilla_r3 = close_4h + (high_4h - low_4h) * 1.1 / 4
    camarilla_s3 = close_4h - (high_4h - low_4h) * 1.1 / 4
    
    # Shift by 1 to use only prior completed 4h bar (no look-ahead)
    camarilla_r3_shifted = np.roll(camarilla_r3, 1)
    camarilla_s3_shifted = np.roll(camarilla_s3, 1)
    camarilla_r3_shifted[0] = np.nan
    camarilla_s3_shifted[0] = np.nan
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3_shifted)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3_shifted)
    
    # Volume confirmation: 20-period EMA of volume on 4h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 + price > 12h EMA34 + volume spike
            if close[i] > camarilla_r3_aligned[i] and close[i] > ema_34_12h_aligned[i] and volume[i] > (1.8 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S3 + price < 12h EMA34 + volume spike
            elif close[i] < camarilla_s3_aligned[i] and close[i] < ema_34_12h_aligned[i] and volume[i] > (1.8 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Camarilla S3 OR price crosses below 12h EMA34
            if close[i] < camarilla_s3_aligned[i] or close[i] < ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Camarilla R3 OR price crosses above 12h EMA34
            if close[i] > camarilla_r3_aligned[i] or close[i] > ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals