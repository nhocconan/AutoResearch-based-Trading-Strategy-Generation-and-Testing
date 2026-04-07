#!/usr/bin/env python3
"""
1h_ema_crossover_4h1d_trend_volume_v1
Hypothesis: On 1-hour timeframe, use EMA(12/26) crossover for entry timing, with trend filter from 4-hour EMA50 and 1-day EMA200. Enter long when fast EMA crosses above slow EMA in uptrend (price > 4h EMA50 and > 1d EMA200) with volume > 1.5x average, short when fast EMA crosses below slow EMA in downtrend with volume confirmation. Exit on opposite crossover. Uses 4h/1d for trend direction, 1h only for entry timing to reduce whipsaw. Designed for 15-37 trades/year to avoid fee drag while capturing trends in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_ema_crossover_4h1d_trend_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for intermediate trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    c_4h = df_4h['close'].values
    ema4h_50 = pd.Series(c_4h).ewm(span=50, adjust=False).mean().values
    ema4h_50_aligned = align_htf_to_ltf(prices, df_4h, ema4h_50)
    
    # Get 1d data for long-term trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    c_1d = df_1d['close'].values
    ema1d_200 = pd.Series(c_1d).ewm(span=200, adjust=False).mean().values
    ema1d_200_aligned = align_htf_to_ltf(prices, df_1d, ema1d_200)
    
    # 1h EMA crossover signals
    ema_fast = pd.Series(close).ewm(span=12, adjust=False).mean().values
    ema_slow = pd.Series(close).ewm(span=26, adjust=False).mean().values
    
    # Volume confirmation: 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if trend filters not available
        if np.isnan(ema4h_50_aligned[i]) or np.isnan(ema1d_200_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Determine trend: need both 4h and 1d uptrend/downtrend
        uptrend_4h = close[i] > ema4h_50_aligned[i]
        uptrend_1d = close[i] > ema1d_200_aligned[i]
        downtrend_4h = close[i] < ema4h_50_aligned[i]
        downtrend_1d = close[i] < ema1d_200_aligned[i]
        
        uptrend = uptrend_4h and uptrend_1d
        downtrend = downtrend_4h and downtrend_1d
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_avg[i] if not np.isnan(vol_avg[i]) else False
        
        # EMA crossover signals
        ema_cross_up = ema_fast[i] > ema_slow[i] and ema_fast[i-1] <= ema_slow[i-1]
        ema_cross_down = ema_fast[i] < ema_slow[i] and ema_fast[i-1] >= ema_slow[i-1]
        
        if position == 1:  # Long position
            # Exit when EMA crosses down
            if ema_cross_down:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit when EMA crosses up
            if ema_cross_up:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Long entry: EMA crosses up in uptrend with volume confirmation
            if ema_cross_up and uptrend and vol_confirm:
                position = 1
                signals[i] = 0.20
            # Short entry: EMA crosses down in downtrend with volume confirmation
            elif ema_cross_down and downtrend and vol_confirm:
                position = -1
                signals[i] = -0.20
    
    return signals