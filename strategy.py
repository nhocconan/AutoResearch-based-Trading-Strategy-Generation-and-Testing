#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_ChandelierExit_EMA13_Trend_Filter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA(13) for trend direction
    close_1d = df_1d['close'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    
    # Calculate ATR(22) for Chandelier Exit
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[0], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=22, adjust=False, min_periods=22).mean().values
    
    # Chandelier Exit long/short levels
    ch_exit_long = pd.Series(high).rolling(window=22, min_periods=22).max().values - 3.0 * atr
    ch_exit_short = pd.Series(low).rolling(window=22, min_periods=22).min().values + 3.0 * atr
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 22  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema13_1d_aligned[i]) or np.isnan(ch_exit_long[i]) or 
            np.isnan(ch_exit_short[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema13_1d_val = ema13_1d_aligned[i]
        long_exit = ch_exit_long[i]
        short_exit = ch_exit_short[i]
        
        if position == 0:
            # Enter long: price above Chandelier long exit + 1d uptrend
            if (close[i] > long_exit and close[i] > ema13_1d_val):
                signals[i] = 0.25
                position = 1
            # Enter short: price below Chandelier short exit + 1d downtrend
            elif (close[i] < short_exit and close[i] < ema13_1d_val):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price below Chandelier long exit
            if close[i] < long_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price above Chandelier short exit
            if close[i] > short_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals