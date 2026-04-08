#!/usr/bin/env python3
# 1h_ema_cross_4h1d_trend_volume
# Hypothesis: EMA crossovers on 1h combined with 4h and 1d trend filters and volume confirmation.
# Long when fast EMA crosses above slow EMA on 1h, price above 4h EMA50 and 1d EMA200, and volume > 1.5x average.
# Short when fast EMA crosses below slow EMA on 1h, price below 4h EMA50 and 1d EMA200, and volume > 1.5x average.
# Exit when opposite EMA crossover occurs or trend filters fail.
# Designed to capture strong trends with confirmation from higher timeframes and volume.
# Target: 60-150 total trades over 4 years (~15-37/year) for 1h timeframe.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_ema_cross_4h1d_trend_volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h and 1d data for trend filters
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1h EMAs for crossover signal
    ema_fast = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema_slow = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1d EMA200 for trend filter
    ema_200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(avg_volume[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: EMA cross down OR trend filters fail
            if (ema_fast[i] < ema_slow[i]) or (close[i] < ema_50_4h_aligned[i]) or (close[i] < ema_200_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: EMA cross up OR trend filters fail
            if (ema_fast[i] > ema_slow[i]) or (close[i] > ema_50_4h_aligned[i]) or (close[i] > ema_200_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # EMA crossover entries
            if (ema_fast[i] > ema_slow[i]) and (close[i] > ema_50_4h_aligned[i]) and (close[i] > ema_200_1d_aligned[i]) and volume_ok:
                position = 1
                signals[i] = 0.20
            elif (ema_fast[i] < ema_slow[i]) and (close[i] < ema_50_4h_aligned[i]) and (close[i] < ema_200_1d_aligned[i]) and volume_ok:
                position = -1
                signals[i] = -0.20
    
    return signals