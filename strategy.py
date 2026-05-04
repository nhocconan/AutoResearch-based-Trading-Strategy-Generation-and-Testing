#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with 1d EMA50 trend filter and volume spike confirmation
# Uses Williams %R(14) from 12h for overbought/oversold signals, 1d EMA50 for trend filter,
# and volume spike for confirmation. Designed for 12-25 trades/year to minimize fee drag.
# Works in bull markets via oversold bounces in uptrend and in bear markets via overbought reversals in downtrend.
# The 1d EMA50 provides a smooth trend filter that avoids whipsaw while capturing major moves.

name = "12h_WilliamsR14_1dEMA50_VolumeSpike_TrendFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for Williams %R calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Williams %R(14) from prior completed 12h bar
    highest_high_14 = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_12h) / (highest_high_14 - lowest_low_14)
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)  # avoid division by zero
    williams_r_shifted = np.roll(williams_r, 1)
    williams_r_shifted[0] = np.nan
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r_shifted)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 trend filter from prior completed 1d bar
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_shifted = np.roll(ema50_1d, 1)
    ema50_1d_shifted[0] = np.nan
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d_shifted)
    
    # Volume confirmation: 20-period EMA of volume on 12h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R oversold (< -80) AND price above 1d EMA50 AND volume spike
            if williams_r_aligned[i] < -80 and close[i] > ema50_1d_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R overbought (> -20) AND price below 1d EMA50 AND volume spike
            elif williams_r_aligned[i] > -20 and close[i] < ema50_1d_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R rises above -50 OR price closes below 1d EMA50
            if williams_r_aligned[i] > -50 or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R falls below -50 OR price closes above 1d EMA50
            if williams_r_aligned[i] < -50 or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals