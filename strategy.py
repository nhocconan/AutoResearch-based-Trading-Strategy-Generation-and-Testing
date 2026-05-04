#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R (14) extreme reversal with 12h EMA50 trend filter and volume spike confirmation
# Williams %R identifies overbought/oversold conditions (< -80 = oversold, > -20 = overbought)
# Reversal from extreme levels with trend alignment and volume confirmation captures mean reversion in ranging markets
# and continuation in trending markets. Uses tighter entry conditions to target 30-60 trades/year, minimizing fee drag.
# Works in both bull and bear markets due to 12h EMA50 trend filter + volume confirmation avoiding false signals.

name = "6h_WilliamsR_14_12hEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 trend filter from prior completed 12h bar
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_shifted = np.roll(ema50_12h, 1)
    ema50_12h_shifted[0] = np.nan
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h_shifted)
    
    # Calculate Williams %R (14) on 6h timeframe
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: 20-period EMA of volume on 6h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R crosses above -80 (oversold reversal) AND 12h EMA50 uptrend AND volume spike
            if williams_r[i] > -80 and williams_r[i-1] <= -80 and close[i] > ema50_12h_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R crosses below -20 (overbought reversal) AND 12h EMA50 downtrend AND volume spike
            elif williams_r[i] < -20 and williams_r[i-1] >= -20 and close[i] < ema50_12h_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses above -20 (overbought) OR price closes below 12h EMA50
            if williams_r[i] > -20 or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses below -80 (oversold) OR price closes above 12h EMA50
            if williams_r[i] < -80 or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals