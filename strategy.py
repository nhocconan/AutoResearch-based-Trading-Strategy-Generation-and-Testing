#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA(34) trend filter and volume confirmation (>1.6x 20 EMA volume)
# Uses Camarilla pivot levels from prior completed 1d bar for structure (breakout = touch of R3/S3 levels)
# 1d EMA(34) filter ensures we only trade in the direction of the higher timeframe trend
# Volume confirmation ensures breakout has sufficient participation
# Discrete sizing 0.25 balances risk and return while minimizing fee churn
# Target: 75-150 total trades over 4 years = 19-38/year for 4h timeframe
# Works in both bull (breakouts continuation) and bear (breakdowns continuation) markets
# BTC/ETH focus: avoids SOL-only bias by requiring HTF trend alignment

name = "4h_Camarilla_R3S3_1dEMA34_VolumeSpike"
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
    
    # Get 1d data for EMA trend filter and Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need enough data for EMA calculation
        return np.zeros(n)
    
    # Calculate 1d EMA(34) trend filter from prior completed 1d bar
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_shifted = np.roll(ema_34_1d, 1)
    ema_34_1d_shifted[0] = np.nan
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d_shifted)
    
    # Calculate Camarilla pivot levels from prior completed 1d bar
    # Camarilla: R4 = close + 1.1*(high-low)*1.1/2, R3 = close + 1.1*(high-low)*1.1/4, etc.
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot levels for each 1d bar
    camarilla_r3 = np.full_like(close_1d, np.nan)
    camarilla_s3 = np.full_like(close_1d, np.nan)
    
    for i in range(len(df_1d)):
        hl_range = high_1d[i] - low_1d[i]
        camarilla_r3[i] = close_1d[i] + (1.1 * hl_range * 1.1) / 4
        camarilla_s3[i] = close_1d[i] - (1.1 * hl_range * 1.1) / 4
    
    # Shift by 1 to use only prior completed 1d bar (no look-ahead)
    camarilla_r3_shifted = np.roll(camarilla_r3, 1)
    camarilla_s3_shifted = np.roll(camarilla_s3, 1)
    camarilla_r3_shifted[0] = np.nan
    camarilla_s3_shifted[0] = np.nan
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_shifted)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_shifted)
    
    # Volume confirmation: 20-period EMA of volume on 4h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price touches or breaks above Camarilla R3 + price > 1d EMA34 + volume spike
            if close[i] >= camarilla_r3_aligned[i] and close[i] > ema_34_1d_aligned[i] and volume[i] > (1.6 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price touches or breaks below Camarilla S3 + price < 1d EMA34 + volume spike
            elif close[i] <= camarilla_s3_aligned[i] and close[i] < ema_34_1d_aligned[i] and volume[i] > (1.6 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Camarilla S3 OR price crosses below 1d EMA34
            if close[i] <= camarilla_s3_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Camarilla R3 OR price crosses above 1d EMA34
            if close[i] >= camarilla_r3_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals