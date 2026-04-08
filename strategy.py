#!/usr/bin/env python3
# 12h_ema50_1d_trend_volume_v1
# Hypothesis: On 12h timeframe, enter long when price crosses above EMA50 with volume > 1.5x average and 1d uptrend; short when price crosses below EMA50 with volume > 1.5x average and 1d downtrend. Exit when price crosses back below/above EMA50 or volume drops below average.
# Uses EMA50 for trend following, volume confirmation for momentum, and 1d trend filter to avoid counter-trend trades. Designed for fewer trades (target 15-30/year) to reduce fee drag and work in both bull and bear markets via trend filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_ema50_1d_trend_volume_v1"
timeframe = "12h"
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
    
    # Calculate EMA50 on 12h
    ema50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Calculate 1d trend filter: EMA50
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    daily_close = df_daily['close'].values
    daily_ema50 = pd.Series(daily_close).ewm(span=50, min_periods=50, adjust=False).mean().values
    daily_ema50_12h = align_htf_to_ltf(prices, df_daily, daily_ema50)
    
    # Volume confirmation: 20-period average on 12h
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(ema50[i]) or np.isnan(daily_ema50_12h[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses back below EMA50 or volume drops below average
            if close[i] < ema50[i] or volume[i] < avg_volume[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses back above EMA50 or volume drops below average
            if close[i] > ema50[i] or volume[i] < avg_volume[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Daily trend filter
            daily_uptrend = close[i] > daily_ema50_12h[i]
            daily_downtrend = close[i] < daily_ema50_12h[i]
            
            # Long entry: price crosses above EMA50 with volume and uptrend
            if close[i] > ema50[i] and close[i-1] <= ema50[i-1] and volume_ok and daily_uptrend:
                position = 1
                signals[i] = 0.25
            # Short entry: price crosses below EMA50 with volume and downtrend
            elif close[i] < ema50[i] and close[i-1] >= ema50[i-1] and volume_ok and daily_downtrend:
                position = -1
                signals[i] = -0.25
    
    return signals