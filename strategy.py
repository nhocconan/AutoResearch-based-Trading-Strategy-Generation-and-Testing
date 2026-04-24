#!/usr/bin/env python3
"""
Hypothesis: 1h mean reversion strategy using 4h Bollinger Bands and 1d RSI regime filter.
- Primary timeframe: 1h targeting 60-150 total trades over 4 years (15-37/year).
- HTF: 4h Bollinger Bands (20,2) for mean reversion zones, 1d RSI(14) for regime filter.
- Entry: Long when price touches 4h lower BB AND 1d RSI < 40 (oversold in bearish regime).
         Short when price touches 4h upper BB AND 1d RSI > 60 (overbought in bullish regime).
- Exit: Opposite BB touch or time-based exit (max 24 bars hold).
- Signal size: 0.20 discrete to minimize fee churn and manage drawdown.
- Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend) via 1d RSI regime.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 4h data for Bollinger Bands
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Get 1d data for RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 4h Bollinger Bands (20,2)
    close_4h = df_4h['close'].values
    ma_4h = pd.Series(close_4h).rolling(window=20, min_periods=20).mean().values
    std_4h = pd.Series(close_4h).rolling(window=20, min_periods=20).std().values
    bb_upper = ma_4h + (2.0 * std_4h)
    bb_lower = ma_4h - (2.0 * std_4h)
    bb_upper_aligned = align_htf_to_ltf(prices, df_4h, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_4h, bb_lower)
    
    # Calculate 1d RSI(14)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    # Start from index where all indicators are ready
    start_idx = max(30, 20, 1)  # 4h BB needs 30, 1d RSI needs 20
    
    for i in range(start_idx, n):
        # Update bars held
        if position != 0:
            bars_since_entry += 1
        
        # Skip if data not ready
        if (np.isnan(bb_upper_aligned[i]) or np.isnan(bb_lower_aligned[i]) or 
            np.isnan(rsi_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        curr_close = close[i]
        in_session = (8 <= hours[i] <= 20)
        
        if position == 0 and in_session:
            # Check for entry signals
            # Long: price touches 4h lower BB AND 1d RSI < 40 (oversold in bearish regime)
            if curr_close <= bb_lower_aligned[i] and rsi_aligned[i] < 40:
                signals[i] = 0.20
                position = 1
                bars_since_entry = 0
            # Short: price touches 4h upper BB AND 1d RSI > 60 (overbought in bullish regime)
            elif curr_close >= bb_upper_aligned[i] and rsi_aligned[i] > 60:
                signals[i] = -0.20
                position = -1
                bars_since_entry = 0
        elif position != 0:
            # Exit conditions: opposite BB touch or max 24 bars hold
            exit_signal = False
            if position == 1 and curr_close >= bb_upper_aligned[i]:
                exit_signal = True  # Long exit: price touches upper BB
            elif position == -1 and curr_close <= bb_lower_aligned[i]:
                exit_signal = True  # Short exit: price touches lower BB
            elif bars_since_entry >= 24:
                exit_signal = True  # Time-based exit
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_BB20_2_RSI14_Regime_MeanRevert_v1"
timeframe = "1h"
leverage = 1.0