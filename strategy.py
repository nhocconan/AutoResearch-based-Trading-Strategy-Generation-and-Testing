#!/usr/bin/env python3
"""
1d_ma_crossover_1w_trend_volume_v1
Hypothesis: On daily timeframe, use dual moving average crossover with weekly trend filter and volume confirmation.
Go long when fast MA crosses above slow MA, weekly trend is up, and volume > 1.5x 20-day average.
Go short when fast MA crosses below slow MA, weekly trend is down, and volume > 1.5x 20-day average.
Exit on opposite crossover. Designed for low-frequency trading (10-30 trades/year) to minimize fee drag while capturing medium-term trends.
Weekly trend filter prevents counter-trend trades during sideways markets. Works in both bull and bear markets by adapting to trend direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_ma_crossover_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Determine weekly trend direction (using EMA slope)
    weekly_trend_up = np.zeros(len(ema_50_1w_aligned), dtype=bool)
    weekly_trend_down = np.zeros(len(ema_50_1w_aligned), dtype=bool)
    for i in range(1, len(ema_50_1w_aligned)):
        if not np.isnan(ema_50_1w_aligned[i]) and not np.isnan(ema_50_1w_aligned[i-1]):
            weekly_trend_up[i] = ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1]
            weekly_trend_down[i] = ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1]
    
    # Calculate daily moving averages
    fast_period = 20
    slow_period = 50
    ma_fast = pd.Series(close).ewm(span=fast_period, adjust=False, min_periods=fast_period).mean().values
    ma_slow = pd.Series(close).ewm(span=slow_period, adjust=False, min_periods=slow_period).mean().values
    
    # Volume filter: 20-day average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(max(50, 20), n):
        # Skip if data not available
        if (np.isnan(ma_fast[i]) or np.isnan(ma_slow[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: fast MA crosses below slow MA
            if ma_fast[i] < ma_slow[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: fast MA crosses above slow MA
            if ma_fast[i] > ma_slow[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only enter with volume confirmation and weekly trend alignment
            if vol_ok:
                # Long: fast MA crosses above slow MA with weekly uptrend
                if (ma_fast[i] > ma_slow[i] and ma_fast[i-1] <= ma_slow[i-1] and 
                    weekly_trend_up[i]):
                    position = 1
                    signals[i] = 0.25
                # Short: fast MA crosses below slow MA with weekly downtrend
                elif (ma_fast[i] < ma_slow[i] and ma_fast[i-1] >= ma_slow[i-1] and 
                      weekly_trend_down[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals