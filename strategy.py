#!/usr/bin/env python3
"""
1h_4h_1d_triple_timeframe_pullback_v1
Hypothesis: Pullback to moving average in trending market with volume confirmation.
Use 1d for trend direction (EMA50), 4h for entry timing (pullback to EMA20), 1h for execution.
Long when: 1d EMA50 up, price pulls back to 4h EMA20 with volume spike.
Short when: 1d EMA50 down, price pulls back to 4h EMA20 with volume spike.
Targets 15-30 trades/year by requiring multi-timeframe alignment + volume filter.
Works in bull (buy pullbacks in uptrend) and bear (sell pullbacks in downtrend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_triple_timeframe_pullback_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend direction
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 4h data for entry timing
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 1d EMA50 for trend
    ema_1d_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_50_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_50)
    
    # 4h EMA20 for pullback entry
    ema_4h_20 = pd.Series(df_4h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_4h_20_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_20)
    
    # Volume confirmation: volume > 2x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > vol_ma * 2.0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(ema_1d_50_aligned[i]) or np.isnan(ema_4h_20_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below 4h EMA20 or 1d trend turns down
            if close[i] < ema_4h_20_aligned[i] or ema_1d_50_aligned[i] < ema_1d_50_aligned[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price crosses above 4h EMA20 or 1d trend turns up
            if close[i] > ema_4h_20_aligned[i] or ema_1d_50_aligned[i] > ema_1d_50_aligned[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: 1d uptrend, price pulls back to 4h EMA20 with volume
            if (ema_1d_50_aligned[i] > ema_1d_50_aligned[i-1] and 
                abs(close[i] - ema_4h_20_aligned[i]) < (high[i] - low[i]) * 0.5 and
                vol_confirm[i]):
                position = 1
                signals[i] = 0.20
            # Short entry: 1d downtrend, price pulls back to 4h EMA20 with volume
            elif (ema_1d_50_aligned[i] < ema_1d_50_aligned[i-1] and 
                  abs(close[i] - ema_4h_20_aligned[i]) < (high[i] - low[i]) * 0.5 and
                  vol_confirm[i]):
                position = -1
                signals[i] = -0.20
    
    return signals