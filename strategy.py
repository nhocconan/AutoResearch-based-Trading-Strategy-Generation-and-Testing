#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R (14) extreme reversal with 1d EMA34 trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions; reversals from extremes work in both bull and bear markets.
# 1d EMA34 ensures trades align with higher timeframe trend, reducing whipsaws.
# Volume spike confirms institutional participation at reversal points.
# Designed for 12-30 trades/year to minimize fee drag on 6h timeframe.

name = "6h_WilliamsR14_1dEMA34_VolumeSpike_TrendFilter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 trend filter from prior completed 1d bar
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_shifted = np.roll(ema34_1d, 1)
    ema34_1d_shifted[0] = np.nan
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d_shifted)
    
    # Calculate Williams %R (14) on 6h timeframe
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: 20-period EMA of volume on 6h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(williams_r[i]) or
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R crosses above -80 FROM BELOW AND 1d EMA34 uptrend AND volume spike
            if (williams_r[i] > -80 and williams_r[i-1] <= -80 and 
                close[i] > ema34_1d_aligned[i] and 
                volume[i] > (2.0 * vol_ema_20[i])):
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R crosses below -20 FROM ABOVE AND 1d EMA34 downtrend AND volume spike
            elif (williams_r[i] < -20 and williams_r[i-1] >= -20 and 
                  close[i] < ema34_1d_aligned[i] and 
                  volume[i] > (2.0 * vol_ema_20[i])):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses above -20 OR price closes below 1d EMA34
            if williams_r[i] > -20 or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses below -80 OR price closes above 1d EMA34
            if williams_r[i] < -80 or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals