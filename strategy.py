#!/usr/bin/env python3
# Hypothesis: 4h Camarilla pivot breakout with 1d EMA34 trend filter and volume spike
# Long when price breaks above S1 with 1d EMA34 upward trend and volume > 2x average
# Short when price breaks below R1 with 1d EMA34 downward trend and volume > 2x average
# Exit when price crosses the 1-day EMA34
# Combines institutional pivot levels with trend alignment and volume confirmation
# Target: 100-180 total trades over 4 years (25-45/year) with size 0.25

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike"
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
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1-day OHLC
    close_1d_prev = df_1d['close'].values
    high_1d_prev = df_1d['high'].values
    low_1d_prev = df_1d['low'].values
    
    # Camarilla multipliers
    R1 = close_1d_prev + 0.183 * (high_1d_prev - low_1d_prev)
    S1 = close_1d_prev - 0.183 * (high_1d_prev - low_1d_prev)
    
    # Align Camarilla levels to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume confirmation: current volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (2.0 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above S1 with upward EMA trend and volume spike
            if (close[i] > S1_aligned[i] and 
                close[i] > ema_34_aligned[i] and  # Additional trend filter
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below R1 with downward EMA trend and volume spike
            elif (close[i] < R1_aligned[i] and 
                  close[i] < ema_34_aligned[i] and  # Additional trend filter
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below EMA34
            if close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above EMA34
            if close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals