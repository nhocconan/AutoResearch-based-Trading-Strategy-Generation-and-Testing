# 4H_CAMARILLA_R1_S1_BREAKOUT_1D_EMA34_VOLUME_SPIKE
# Hypothesis: Camarilla pivot levels (R1/S1) from 1d combined with EMA34 trend filter and volume spike confirmation
# Captures institutional breakouts while filtering chop. Works in both bull (break above R1 in uptrend) and bear (break below S1 in downtrend).
# Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years).

name = "4H_CAMARILLA_R1_S1_BREAKOUT_1D_EMA34_VOLUME_SPIKE"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for previous day
    # Using previous day's high, low, close to avoid look-ahead
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    r1 = close_1d + 1.1 * range_1d / 12
    s1 = close_1d - 1.1 * range_1d / 12
    
    # EMA34 for trend filter
    ema34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    # Align to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 1  # Need at least one day of data
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(volume_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Close breaks above R1 in uptrend with volume spike
            if (close[i] > r1_aligned[i] and 
                close[i] > ema34_aligned[i] and 
                volume_spike_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Close breaks below S1 in downtrend with volume spike
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema34_aligned[i] and 
                  volume_spike_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close falls back below EMA34 or opposite S1 break
            if (close[i] <= ema34_aligned[i] or 
                close[i] < s1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close rises back above EMA34 or opposite R1 break
            if (close[i] >= ema34_aligned[i] or 
                close[i] > r1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals