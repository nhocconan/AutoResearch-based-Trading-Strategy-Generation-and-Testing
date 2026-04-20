#!/usr/bin/env python3
# 12h_Daily_Close_Breakout_RSI_Filter
# Hypothesis: Breakouts from daily close with RSI filter on 12h timeframe.
# Uses daily close as dynamic support/resistance. RSI(14) filters for momentum strength.
# Works in bull markets (breakouts up) and bear markets (breakdowns down).
# Target: 20-40 trades per year to avoid fee drag.

name = "12h_Daily_Close_Breakout_RSI_Filter"
timeframe = "12h"
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
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily close as support/resistance
    daily_close = df_1d['close'].values
    daily_close_aligned = align_htf_to_ltf(prices, df_1d, daily_close)
    
    # RSI(14) on 12h closes
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation (20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(daily_close_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout above daily close with RSI > 50 and volume
            if (close[i] > daily_close_aligned[i] * 1.002 and 
                rsi[i] > 50 and 
                volume[i] > 1.5 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short breakdown below daily close with RSI < 50 and volume
            elif (close[i] < daily_close_aligned[i] * 0.998 and 
                  rsi[i] < 50 and 
                  volume[i] > 1.5 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: breakdown below daily close
            if close[i] < daily_close_aligned[i] * 0.998:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: breakout above daily close
            if close[i] > daily_close_aligned[i] * 1.002:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals