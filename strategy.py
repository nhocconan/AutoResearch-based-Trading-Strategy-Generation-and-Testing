#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h 4h/1d multi-timeframe strategy using 4h trend (EMA50) and 1d support/resistance (EMA200)
# Long when 1h price > 4h EMA50 and > 1d EMA200, short when < 4h EMA50 and < 1d EMA200
# Uses session filter (08-20 UTC) to avoid low-liquidity hours
# Fixed position size 0.20 to limit drawdown and control trade frequency
# Designed for low turnover (<150 trades/year) to minimize fee drag in choppy markets

name = "1h_4hEMA50_1dEMA200_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Get 1d data for long-term trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate 4h EMA50 trend
    ema50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate 1d EMA200 trend
    ema200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Session filter: 08-20 UTC (avoid low-liquidity hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Start after warmup period for indicators
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if any data is NaN or outside session
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or 
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Long condition: price above both 4h EMA50 and 1d EMA200
        if close[i] > ema50_4h_aligned[i] and close[i] > ema200_1d_aligned[i]:
            signals[i] = 0.20
        # Short condition: price below both 4h EMA50 and 1d EMA200
        elif close[i] < ema50_4h_aligned[i] and close[i] < ema200_1d_aligned[i]:
            signals[i] = -0.20
        # Otherwise flat
        else:
            signals[i] = 0.0
    
    return signals