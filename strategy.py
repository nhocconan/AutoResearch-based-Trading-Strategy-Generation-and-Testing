#!/usr/bin/env python3
# Hypothesis: 1h trend-following strategy using 4h EMA20 and 1d EMA50 for trend confirmation, with volume spike filter
# Long when price > 4h EMA20 > 1d EMA50 and volume > 1.5x 20-period average
# Short when price < 4h EMA20 < 1d EMA50 and volume > 1.5x 20-period average
# Exit when price crosses back below/above 4h EMA20
# Uses multi-timeframe alignment to avoid look-ahead bias and session filter (08-20 UTC) to reduce noise
# Target: 60-150 total trades over 4 years (15-37/year) with size 0.20

name = "1h_EMA20_EMA50_Trend_Volume"
timeframe = "1h"
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
    
    # Calculate 4h EMA20 for trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    ema20_4h = pd.Series(df_4h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema20_4h_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_confirm[i]) or not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price > 4h EMA20 > 1d EMA50 and volume spike
            if (close[i] > ema20_4h_aligned[i] and 
                ema20_4h_aligned[i] > ema50_1d_aligned[i] and
                vol_confirm[i]):
                signals[i] = 0.20
                position = 1
            # Enter short: price < 4h EMA20 < 1d EMA50 and volume spike
            elif (close[i] < ema20_4h_aligned[i] and 
                  ema20_4h_aligned[i] < ema50_1d_aligned[i] and
                  vol_confirm[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below 4h EMA20
            if close[i] < ema20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price crosses above 4h EMA20
            if close[i] > ema20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals