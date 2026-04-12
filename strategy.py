#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1w EMA34 for HTF trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all indicators to LTF (1d)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(ema34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend conditions
        bullish_trend = close[i] > ema50_1d_aligned[i] and close[i] > ema34_1w_aligned[i]
        bearish_trend = close[i] < ema50_1d_aligned[i] and close[i] < ema34_1w_aligned[i]
        
        # Entry logic: Strong trend alignment
        long_entry = bullish_trend and position != 1
        short_entry = bearish_trend and position != -1
        
        # Exit logic: Trend reversal
        long_exit = not bullish_trend
        short_exit = not bearish_trend
        
        if long_entry:
            position = 1
            signals[i] = 0.25
        elif short_entry:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_ema50_ema34_trend_filter_v1"
timeframe = "1d"
leverage = 1.0