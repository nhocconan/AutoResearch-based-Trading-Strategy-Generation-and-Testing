#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d trend filter and volume confirmation
# Elder Ray uses EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Long when Bull Power > 0 and Bear Power rising (less negative) with 1d uptrend and volume spike
# Short when Bear Power < 0 and Bull Power falling (less positive) with 1d downtrend and volume spike
# Designed for low-frequency, high-conviction trades to work in both bull and bear markets via trend alignment.

name = "6h_ElderRay_1dTrend_Volume"
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
    
    # Calculate EMA13 for Elder Ray
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Get 1d data for trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike (2x 20-period EMA)
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure EMA34 has enough data
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Bull Power > 0, Bear Power rising (less negative), 1d uptrend, volume spike
            if (bull_power[i] > 0 and 
                bear_power[i] > bear_power[i-1] and 
                close[i] > ema34_1d_aligned[i] and vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Bear Power < 0, Bull Power falling (less positive), 1d downtrend, volume spike
            elif (bear_power[i] < 0 and 
                  bull_power[i] < bull_power[i-1] and 
                  close[i] < ema34_1d_aligned[i] and vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power <= 0 or trend fails
            if (bull_power[i] <= 0 or 
                close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power >= 0 or trend fails
            if (bear_power[i] >= 0 or 
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals