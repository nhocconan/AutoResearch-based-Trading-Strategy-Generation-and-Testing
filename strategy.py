#!/usr/bin/env python3
# 12h_kama_ma_crossover_1w_trend_volume
# Hypothesis: KAMA (2-period EMA) crossover on 12h with 1-week EMA trend filter and volume confirmation.
# Long when KAMA fast > KAMA slow with uptrend (price > 1w EMA50) and volume > 1.5x average.
# Short when KAMA fast < KAMA slow with downtrend (price < 1w EMA50) and volume > 1.5x average.
# Exit when KAMA lines cross in opposite direction.
# Designed to capture trends with adaptive smoothing in both bull and bear markets.
# Target: 50-150 total trades over 4 years (~12-37/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_kama_ma_crossover_1w_trend_volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on 12h data
    # Fast EMA (2-period) and Slow EMA (30-period) as proxies for KAMA responsiveness
    close_series = pd.Series(close)
    ema_fast = close_series.ewm(span=2, adjust=False, min_periods=2).mean().values
    ema_slow = close_series.ewm(span=30, adjust=False, min_periods=30).mean().values
    
    # Calculate average volume for confirmation (30-period)
    avg_volume = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(avg_volume[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: KAMA fast crosses below KAMA slow
            if ema_fast[i] < ema_slow[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: KAMA fast crosses above KAMA slow
            if ema_fast[i] > ema_slow[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # KAMA crossover entries
            if (ema_fast[i] > ema_slow[i]) and (close[i] > ema_50_1w_aligned[i]) and volume_ok:
                position = 1
                signals[i] = 0.25
            elif (ema_fast[i] < ema_slow[i]) and (close[i] < ema_50_1w_aligned[i]) and volume_ok:
                position = -1
                signals[i] = -0.25
    
    return signals