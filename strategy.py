#!/usr/bin/env python3
name = "1d_1Week_VolatilityBreakout_TrendFilter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for ATR and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly ATR (14-period)
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Weekly EMA20 for trend
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align to daily
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Daily volatility breakout
    # Calculate daily range
    daily_range = high - low
    # 20-day average range
    avg_range = pd.Series(daily_range).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(20, 20)  # For daily range and weekly indicators
    
    for i in range(start_idx, n):
        if np.isnan(atr_1w_aligned[i]) or np.isnan(ema20_1w_aligned[i]) or np.isnan(avg_range[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: daily range > 1.5 * weekly ATR AND price > weekly EMA20
            if daily_range[i] > 1.5 * atr_1w_aligned[i] and close[i] > ema20_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: daily range > 1.5 * weekly ATR AND price < weekly EMA20
            elif daily_range[i] > 1.5 * atr_1w_aligned[i] and close[i] < ema20_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below weekly EMA20
            if close[i] < ema20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above weekly EMA20
            if close[i] > ema20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals