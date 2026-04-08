# 1h_4h_1d_trend_following_v2
# Hypothesis: Use 4h EMA trend filter and 1d trend confirmation with 1h breakout entries to capture momentum in both bull and bear markets. Focus on reducing trade frequency to 15-30/year by requiring multi-timeframe alignment and volume confirmation. 1h timeframe provides timely entries while 4h/1d filters prevent countertrend trades.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_trend_following_v2"
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
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Get 1d data for trend confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 4h close
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate EMA200 on 1d close for long-term trend
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Volume confirmation: 1h volume > 1.8x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > vol_ma * 1.8
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(ema50_4h_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below 4h EMA50
            if close[i] < ema50_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price closes above 4h EMA50
            if close[i] > ema50_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price above 4h EMA50 AND above 1d EMA200, with volume confirmation
            if close[i] > ema50_4h_aligned[i] and close[i] > ema200_1d_aligned[i] and vol_confirm[i]:
                position = 1
                signals[i] = 0.20
            # Short entry: price below 4h EMA50 AND below 1d EMA200, with volume confirmation
            elif close[i] < ema50_4h_aligned[i] and close[i] < ema200_1d_aligned[i] and vol_confirm[i]:
                position = -1
                signals[i] = -0.20
    
    return signals