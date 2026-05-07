#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_12hEMA34_VolumeSpike
# Hypothesis: Camarilla pivot levels act as dynamic support/resistance. Breaking above R1 or below S1 with
# volume confirmation and 12h EMA34 trend filter captures institutional breakout moves. Works in bull
# markets (long on R1 break + uptrend) and bear markets (short on S1 break + downtrend). Volume spike
# filters low-conviction moves. Target: 20-40 trades/year per symbol.

name = "4h_Camarilla_R1_S1_Breakout_12hEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels (R1, S1) from previous day
    # R1 = close + 1.1 * (high - low) / 12
    # S1 = close - 1.1 * (high - low) / 12
    rng = high_1d - low_1d
    r1 = close_1d + 1.1 * rng / 12
    s1 = close_1d - 1.1 * rng / 12
    
    # Align pivot levels to 4h timeframe (use previous day's levels)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) == 0:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate EMA34 for 12h trend filter
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Volume spike detection: 2.0x average volume (40-period = ~3.33 days on 4h chart)
    vol_ma = pd.Series(volume).rolling(window=40, min_periods=40).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 40)  # Ensure we have EMA34 and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema34_12h_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1, price above 12h EMA34 (uptrend), volume spike
            if (high[i] > r1_aligned[i-1] and 
                close[i] > ema34_12h_aligned[i] and 
                volume[i] > 2.0 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1, price below 12h EMA34 (downtrend), volume spike
            elif (low[i] < s1_aligned[i-1] and 
                  close[i] < ema34_12h_aligned[i] and 
                  volume[i] > 2.0 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below S1 OR price crosses below 12h EMA34
            if (low[i] < s1_aligned[i-1] or close[i] < ema34_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above R1 OR price crosses above 12h EMA34
            if (high[i] > r1_aligned[i-1] or close[i] > ema34_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals