#!/usr/bin/env python3
"""
1h_4d_ema_pullback
Trades pullbacks to EMA20 in 1h timeframe, aligned with 4h trend (EMA50) and 1d momentum filter (close > open).
Long when: 4h EMA50 up, price pulls back to 1h EMA20 from above, and 1d bullish candle.
Short when: 4h EMA50 down, price pulls back to 1h EMA20 from below, and 1d bearish candle.
Exit when price crosses EMA20 in opposite direction.
Designed for low frequency: ~20-40 trades/year by requiring 4h trend + 1d filter + pullback precision.
"""

name = "1h_4d_ema_pullback"
timeframe = "1h"
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
    
    # 4h EMA50 for trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 1d candle direction for momentum filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    open_1d = df_1d['open'].values
    close_1d = df_1d['close'].values
    bullish_1d = close_1d > open_1d  # True if bullish daily candle
    bearish_1d = close_1d < open_1d  # True if bearish daily candle
    bullish_1d_aligned = align_htf_to_ltf(prices, df_1d, bullish_1d.astype(float))
    bearish_1d_aligned = align_htf_to_ltf(prices, df_1d, bearish_1d.astype(float))
    
    # 1h EMA20 for pullback entries
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(bullish_1d_aligned[i]) or 
            np.isnan(bearish_1d_aligned[i]) or np.isnan(ema20[i])):
            signals[i] = 0.0
            continue
        
        # Determine 4h trend: comparing current to previous EMA50
        if i == 100:
            prev_ema50 = ema50_4h_aligned[i-1]
        else:
            prev_ema50 = ema50_4h_aligned[i-1]
        curr_ema50 = ema50_4h_aligned[i]
        uptrend_4h = curr_ema50 > prev_ema50
        downtrend_4h = curr_ema50 < prev_ema50
        
        # Long: 4h uptrend, pullback to EMA20 from above, 1d bullish
        if (uptrend_4h and low[i] <= ema20[i] and close[i] > ema20[i] and 
            bullish_1d_aligned[i] > 0.5 and position != 1):
            position = 1
            signals[i] = 0.20
        # Short: 4h downtrend, pullback to EMA20 from below, 1d bearish
        elif (downtrend_4h and high[i] >= ema20[i] and close[i] < ema20[i] and 
              bearish_1d_aligned[i] > 0.5 and position != -1):
            position = -1
            signals[i] = -0.20
        # Exit: price crosses EMA20 in opposite direction
        elif position == 1 and close[i] < ema20[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > ema20[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals