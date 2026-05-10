#!/usr/bin/env python3
# 6h_Trend_Stack_With_1dVWAP
# Hypothesis: Stack multiple trend signals (6h EMA10/50, 1d EMA50, 1d VWAP) to capture strong trends in both bull and bear markets.
# Long when price > EMA10 > EMA50 (6h) AND price > EMA50 (1d) AND price > VWAP (1d).
# Short when price < EMA10 < EMA50 (6h) AND price < EMA50 (1d) AND price < VWAP (1d).
# Uses volume-weighted average price (VWAP) as a dynamic support/resistance filter from higher timeframe.
# Targets 20-40 trades/year by requiring confluence of multiple timeframe trends.

name = "6h_Trend_Stack_With_1dVWAP"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for VWAP and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 6h EMA10 and EMA50 for trend
    ema_10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).values
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).values
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Daily VWAP calculation (typical price * volume)
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    vwap = (typical_price * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap = vwap.values
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(10, 50, 50)  # Warmup for EMAs
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_10[i]) or np.isnan(ema_50[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vwap_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend alignment conditions
        bullish_stack = (close[i] > ema_10[i]) and (ema_10[i] > ema_50[i]) and \
                        (close[i] > ema_50_1d_aligned[i]) and (close[i] > vwap_aligned[i])
        bearish_stack = (close[i] < ema_10[i]) and (ema_10[i] < ema_50[i]) and \
                        (close[i] < ema_50_1d_aligned[i]) and (close[i] < vwap_aligned[i])
        
        if position == 0:
            # Long entry: full bullish stack
            if bullish_stack:
                signals[i] = 0.25
                position = 1
            # Short entry: full bearish stack
            elif bearish_stack:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: break of any trend layer
            if not bullish_stack:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: break of any trend layer
            if not bearish_stack:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals