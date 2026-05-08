#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h1d_Trend_1hEntry_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    # 4h EMA50 for trend
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Get 1d data for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 1d EMA200 for long-term trend
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # 1h ATR for stop loss (not used in signal but for reference)
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr1 = np.maximum(tr1, np.abs(low[1:] - close[:-1]))
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], tr1])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(volume[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 4h EMA50 > 1d EMA200 (bullish regime) + price > 4h EMA50
            if ema50_4h_aligned[i] > ema200_1d_aligned[i] and close[i] > ema50_4h_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: 4h EMA50 < 1d EMA200 (bearish regime) + price < 4h EMA50
            elif ema50_4h_aligned[i] < ema200_1d_aligned[i] and close[i] < ema50_4h_aligned[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price < 4h EMA50 OR regime change (4h EMA50 < 1d EMA200)
            if close[i] < ema50_4h_aligned[i] or ema50_4h_aligned[i] < ema200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price > 4h EMA50 OR regime change (4h EMA50 > 1d EMA200)
            if close[i] > ema50_4h_aligned[i] or ema50_4h_aligned[i] > ema200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals