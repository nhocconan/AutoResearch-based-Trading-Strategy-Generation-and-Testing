#!/usr/bin/env python3
"""
1h_SR_SR_Filter
Simple support/resistance with 4h/1d context:
- Long when price > 4h EMA20 + 1h price > 4h support level (from prior 4h candle low)
- Short when price < 4h EMA20 + 1h price < 4h resistance level (from prior 4h candle high)
- Exit when price crosses back through 4h EMA20
- Uses 1d trend filter: only long if 1d close > 1d EMA50, only short if 1d close < 1d EMA50
- Session filter: 08-20 UTC only
- Fixed position size: 0.20
Designed for 15-35 trades/year per symbol
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 4h data for EMA and support/resistance levels
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h EMA20
    close_4h_series = pd.Series(close_4h)
    ema20_4h = close_4h_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # 4h support/resistance levels (prior candle low/high)
    support_4h = np.roll(low_4h, 1)  # prior candle low
    resistance_4h = np.roll(high_4h, 1)  # prior candle high
    support_4h[0] = np.nan
    resistance_4h[0] = np.nan
    
    # Align 4h indicators to 1h
    ema20_4h_1h = align_htf_to_ltf(prices, df_4h, ema20_4h)
    support_4h_1h = align_htf_to_ltf(prices, df_4h, support_4h)
    resistance_4h_1h = align_htf_to_ltf(prices, df_4h, resistance_4h)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 1h
    ema50_1d_1h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need sufficient data for EMA calculations
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not session_mask[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is not available
        if (np.isnan(ema20_4h_1h[i]) or np.isnan(support_4h_1h[i]) or 
            np.isnan(resistance_4h_1h[i]) or np.isnan(ema50_1d_1h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price > 4h EMA20 AND price > 4h support AND 1d uptrend
            if (close[i] > ema20_4h_1h[i] and 
                close[i] > support_4h_1h[i] and 
                close[i] > ema50_1d_1h[i]):
                signals[i] = 0.20
                position = 1
            # Short: price < 4h EMA20 AND price < 4h resistance AND 1d downtrend
            elif (close[i] < ema20_4h_1h[i] and 
                  close[i] < resistance_4h_1h[i] and 
                  close[i] < ema50_1d_1h[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below 4h EMA20
            if close[i] < ema20_4h_1h[i]:
                signals[i] = -0.20  # reverse to short
                position = -1
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price crosses above 4h EMA20
            if close[i] > ema20_4h_1h[i]:
                signals[i] = 0.20  # reverse to long
                position = 1
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_SR_SR_Filter"
timeframe = "1h"
leverage = 1.0