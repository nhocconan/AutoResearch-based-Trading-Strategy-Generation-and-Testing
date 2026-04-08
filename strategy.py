#!/usr/bin/env python3
# 1h_4h_1d_ema_trend_with_volume
# Hypothesis: Use 1d EMA for long-term trend direction, 4h EMA for medium-term trend confirmation,
# and 1h for entry timing with volume confirmation. Trades only in direction of higher timeframe
# trends to avoid whipsaw. Volume filter ensures momentum behind moves. Designed for 1h timeframe
# to target 15-35 trades/year by requiring multi-timeframe alignment and volume confirmation.
# Works in bull markets (trend following) and bear markets (counter-trend bounces when 1d trend
# is weak but 4h/1d alignment occurs).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_ema_trend_with_volume"
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
    
    # Get 4h data for medium-term trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Get 1d data for long-term trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA(21) for medium-term trend
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 1d EMA(50) for long-term trend
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: volume > 1.8x average of last 24 periods (1 day)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_confirm = volume > vol_ma * 1.8
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below 4h EMA or loses upward momentum
            if close[i] < ema_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price closes above 4h EMA or loses downward momentum
            if close[i] > ema_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price above both EMAs with volume confirmation
            if (close[i] > ema_4h_aligned[i] and 
                close[i] > ema_1d_aligned[i] and 
                vol_confirm[i]):
                position = 1
                signals[i] = 0.20
            # Short entry: price below both EMAs with volume confirmation
            elif (close[i] < ema_4h_aligned[i] and 
                  close[i] < ema_1d_aligned[i] and 
                  vol_confirm[i]):
                position = -1
                signals[i] = -0.20
    
    return signals