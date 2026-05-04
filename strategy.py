#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume confirmation (>1.5x 20 EMA volume)
# Uses Williams Alligator (Jaw=TEETH=13, Teeth=TEETH=8, Lips=TEETH=5) from 1d timeframe for structure
# 12h EMA50 filter ensures we trade in direction of higher timeframe trend (avoids counter-trend whipsaws)
# Volume confirmation ensures breakout has sufficient participation (>1.5x average volume)
# Discrete sizing 0.25 balances risk and return while minimizing fee churn
# Target: 50-150 total trades over 4 years = 12-37/year for 12h timeframe
# Works in both bull (Alligator alignment up) and bear (Alligator alignment down) markets
# Focus on BTC/ETH by requiring 12h trend alignment (avoids SOL-only bias, more robust across regimes)

name = "12h_Williams_Alligator_12hEMA50_VolumeConfirm"
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
    
    # Get 1d data for Williams Alligator calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:  # Need at least 13 days for Alligator calculation
        return np.zeros(n)
    
    median_price_1d = (df_1d['high'].values + df_1d['low'].values) / 2.0
    
    # Williams Alligator: Jaw (13,8), Teeth (8,5), Lips (5,3) - all SMMA
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        # First value is simple SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (PERIOD-1) + CURRENT_VALUE) / PERIOD
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(median_price_1d, 13)  # Blue line
    teeth = smma(median_price_1d, 8)  # Red line
    lips = smma(median_price_1d, 5)   # Green line
    
    # Shift by 1 to use only prior completed 1d bar (no look-ahead)
    jaw_shifted = np.roll(jaw, 1)
    teeth_shifted = np.roll(teeth, 1)
    lips_shifted = np.roll(lips, 1)
    jaw_shifted[0] = np.nan
    teeth_shifted[0] = np.nan
    lips_shifted[0] = np.nan
    
    # Align Alligator lines to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_shifted)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_shifted)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_shifted)
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need enough data for EMA calculation
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA(50) trend filter from prior completed 12h bar
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_shifted = np.roll(ema_50_12h, 1)
    ema_50_12h_shifted[0] = np.nan
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h_shifted)
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Alligator aligned up (Lips > Teeth > Jaw) AND price > 12h EMA50 AND volume spike
            if lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i] and \
               close[i] > ema_50_12h_aligned[i] and volume[i] > (1.5 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Alligator aligned down (Lips < Teeth < Jaw) AND price < 12h EMA50 AND volume spike
            elif lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i] and \
                 close[i] < ema_50_12h_aligned[i] and volume[i] > (1.5 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator alignment breaks down OR price crosses below 12h EMA50
            if lips_aligned[i] <= teeth_aligned[i] or teeth_aligned[i] <= jaw_aligned[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator alignment breaks up OR price crosses above 12h EMA50
            if lips_aligned[i] >= teeth_aligned[i] or teeth_aligned[i] >= jaw_aligned[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals