#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h EMA200 + Volume Spike + 1d Trend Filter
# Long when price > EMA200, volume > 2x average, and 1d uptrend
# Short when price < EMA200, volume > 2x average, and 1d downtrend
# EMA200 defines long-term trend, volume spike confirms momentum, 1d filter ensures alignment
# Targets 60-150 total trades over 4 years (15-37/year) to balance opportunity and cost

name = "4h_EMA200_Volume_1dTrend"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data once for EMA200
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 200:
        return np.zeros(n)
    
    # Calculate EMA200 on 4h close
    close_4h = df_4h['close'].values
    ema200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema200_4h)
    
    # Get 1d data once for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike: current volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # warmup for EMA200
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema200_4h_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema200_val = ema200_4h_aligned[i]
        ema50_1d_val = ema50_1d_aligned[i]
        vol_spike_val = vol_spike[i]
        close_val = close[i]
        
        if position == 0:
            # Enter long: price > EMA200, volume spike, 1d uptrend
            if close_val > ema200_val and vol_spike_val and ema50_1d_val > 0:
                signals[i] = 0.20
                position = 1
            # Enter short: price < EMA200, volume spike, 1d downtrend
            elif close_val < ema200_val and vol_spike_val and ema50_1d_val < 0:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price < EMA200 or 1d trend down
            if close_val < ema200_val or ema50_1d_val < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price > EMA200 or 1d trend up
            if close_val > ema200_val or ema50_1d_val > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals