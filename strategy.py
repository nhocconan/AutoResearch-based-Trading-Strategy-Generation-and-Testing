#!/usr/bin/env python3

"""
Hypothesis: 12-hour Donchian channel breakout with weekly trend filter and volume confirmation.
This strategy captures trend continuation in strong moves while avoiding whipsaws in ranging markets.
The weekly EMA filter ensures we only trade in the direction of the higher timeframe trend,
and volume spikes confirm institutional participation. Target: 15-25 trades/year per symbol.
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
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA20 for trend filter
    close_1w = pd.Series(df_1w['close'].values)
    ema20_1w = close_1w.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Calculate 12-hour Donchian channel (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12-period volume average for spike detection
    vol_avg_12 = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(ema20_1w_aligned[i]) or np.isnan(vol_avg_12[i])):
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
            # Long: Price breaks above Donchian upper band with bullish weekly trend and volume spike
            if (close[i] > high_max[i] and 
                close[i] > ema20_1w_aligned[i] and  # Bullish weekly trend: price above weekly EMA20
                volume[i] > 1.8 * vol_avg_12[i]):  # Volume spike
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower band with bearish weekly trend and volume spike
            elif (close[i] < low_min[i] and 
                  close[i] < ema20_1w_aligned[i] and  # Bearish weekly trend: price below weekly EMA20
                  volume[i] > 1.8 * vol_avg_12[i]):  # Volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price returns to opposite Donchian band
            exit_signal = False
            
            if position == 1:
                # Exit long: price returns to or below Donchian lower band
                if close[i] <= low_min[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price returns to or above Donchian upper band
                if close[i] >= high_max[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian_20_1wEMA20_Trend_Volume"
timeframe = "12h"
leverage = 1.0