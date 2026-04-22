#!/usr/bin/env python3

"""
Hypothesis: 4-hour Camarilla pivot (R1/S1) breakout with 1-week trend filter and volume confirmation.
Camarilla levels provide high-probability reversal/breakout points. Weekly trend filter ensures
trades align with higher timeframe direction. Volume spikes confirm institutional interest.
This combination works in both bull and bear markets by adapting to weekly trend while
capitalizing on mean-reversion breakouts at key levels. Target: 20-40 trades/year per symbol.
"""

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
    
    # Load weekly data - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly EMA for trend filter (20-period)
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align weekly EMA to 4h timeframe
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla multipliers
    R1 = close_1d + 1.1 * (high_1d - low_1d) / 12
    S1 = close_1d - 1.1 * (high_1d - low_1d) / 12
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Calculate 4h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1, above weekly EMA, volume spike
            if (close[i] > R1_aligned[i] and    # Break above R1
                close[i] > ema_20_1w_aligned[i] and # Above weekly EMA (bullish trend)
                volume[i] > 2.0 * vol_avg_20[i]): # Volume spike
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1, below weekly EMA, volume spike
            elif (close[i] < S1_aligned[i] and   # Break below S1
                  close[i] < ema_20_1w_aligned[i] and # Below weekly EMA (bearish trend)
                  volume[i] > 2.0 * vol_avg_20[i]): # Volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to midpoint of previous day's range
            midpoint = (high_1d[i-1] + low_1d[i-1]) / 2.0 if i > 0 else close[i]
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below midpoint
                if close[i] < midpoint:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above midpoint
                if close[i] > midpoint:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1wEMA20_Volume"
timeframe = "4h"
leverage = 1.0