#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d trend filter and volume spike
# Elder Ray measures bull/bear power via EMA(13) and price extremes.
# Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
# Long when Bull Power > 0 and increasing + 1d uptrend + volume spike
# Short when Bear Power < 0 and decreasing + 1d downtrend + volume spike
# Designed for low-frequency trades (<150 total) to minimize fee drag and capture
# institutional buying/selling pressure in both bull and bear markets.

name = "6h_ElderRay_1dEMA13_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 13-period EMA for Elder Ray on 6h data
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Bull Power and Bear Power
    bull_power = high - ema13  # Higher = stronger bullish pressure
    bear_power = low - ema13   # Lower (more negative) = stronger bearish pressure
    
    # Calculate 1d EMA13 for trend filter
    close_1d = df_1d['close'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    
    # Volume spike (2.0x 20-period EMA)
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure EMA13 has enough data
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema13_1d_aligned[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Bull Power positive AND rising + 1d uptrend + volume spike
            if (bull_power[i] > 0 and bull_power[i] > bull_power[i-1] and 
                close[i] > ema13_1d_aligned[i] and vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Bear Power negative AND falling + 1d downtrend + volume spike
            elif (bear_power[i] < 0 and bear_power[i] < bear_power[i-1] and 
                  close[i] < ema13_1d_aligned[i] and vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power turns negative or trend fails
            if (bull_power[i] <= 0 or close[i] < ema13_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power turns positive or trend fails
            if (bear_power[i] >= 0 or close[i] > ema13_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals