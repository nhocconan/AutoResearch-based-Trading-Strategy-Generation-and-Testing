#!/usr/bin/env python3
# 6h_1d_1w_triple_timeframe_ema_trend_volume_v1
# Hypothesis: Use 1d EMA(50) for medium-term trend, 1w EMA(20) for long-term trend, and 60-period EMA on 6h for entry timing.
# Enter long when 6h EMA crosses above 1d EMA and price > 1w EMA; short when 6h EMA crosses below 1d EMA and price < 1w EMA.
# Require volume > 1.5x 20-period average for institutional confirmation.
# Works in bull markets (trend following) and bear markets (counter-trend reversals from overextended levels).
# Target: 15-25 trades/year per symbol (60-100 total over 4 years) by requiring multi-timeframe alignment and volume filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_1w_triple_timeframe_ema_trend_volume_v1"
timeframe = "6h"
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
    
    # Get 6h data for EMA calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 60:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1w data for long-term trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 60-period EMA on 6h for entry timing
    close_6h = df_6h['close'].values
    ema_60_6h = pd.Series(close_6h).ewm(span=60, adjust=False, min_periods=60).mean().values
    ema_60_6h_aligned = align_htf_to_ltf(prices, df_6h, ema_60_6h)
    
    # Calculate 50-period EMA on 1d for medium-term trend
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 20-period EMA on 1w for long-term trend
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume confirmation: volume > 1.5x average of last 20 periods (~5 days in 6h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(ema_60_6h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or \
           np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: 6h EMA crosses below 1d EMA or price falls below 1w EMA
            if ema_60_6h_aligned[i] < ema_50_1d_aligned[i] or close[i] < ema_20_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: 6h EMA crosses above 1d EMA or price rises above 1w EMA
            if ema_60_6h_aligned[i] > ema_50_1d_aligned[i] or close[i] > ema_20_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: 6h EMA crosses above 1d EMA with price above 1w EMA and volume
            if (ema_60_6h_aligned[i] > ema_50_1d_aligned[i] and 
                close[i] > ema_20_1w_aligned[i] and
                vol_confirm[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: 6h EMA crosses below 1d EMA with price below 1w EMA and volume
            elif (ema_60_6h_aligned[i] < ema_50_1d_aligned[i] and 
                  close[i] < ema_20_1w_aligned[i] and
                  vol_confirm[i]):
                position = -1
                signals[i] = -0.25
    
    return signals