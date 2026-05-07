#!/usr/bin/env python3
name = "6h_TRIX_Signal_Crossover_12hTrend"
timeframe = "6h"
leverage = 1.0

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
    
    # 12h trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    trend_up = close > ema34_12h_aligned
    trend_down = close < ema34_12h_aligned
    
    # TRIX calculation (15-period triple EMA)
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    
    # TRIX = (ema3 - ema3_prev) / ema3_prev * 100
    trix = np.zeros(n)
    trix[1:] = (ema3[1:] - ema3[:-1]) / ema3[:-1] * 100
    trix[0] = 0
    
    # Signal line (9-period EMA of TRIX)
    signal_line = pd.Series(trix).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # Histogram (TRIX - signal)
    hist = trix - signal_line
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 2  # ~12 hours
    
    start_idx = max(15*3, 9)  # Ensure TRIX calculation is valid
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema34_12h_aligned[i]) or 
            np.isnan(trix[i]) or 
            np.isnan(signal_line[i]) or 
            np.isnan(hist[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: TRIX crosses above signal line AND 12h uptrend
            if hist[i-1] <= 0 and hist[i] > 0 and trend_up[i]:
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: TRIX crosses below signal line AND 12h downtrend
            elif hist[i-1] >= 0 and hist[i] < 0 and trend_down[i]:
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: TRIX crosses below signal line OR trend turns down
            if hist[i] < 0 or not trend_up[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: TRIX crosses above signal line OR trend turns up
            if hist[i] > 0 or not trend_down[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: TRIX (triple exponential average) momentum oscillator with signal line crossovers
# and 12h trend filter captures momentum shifts in both bull and bear markets.
# Long when TRIX crosses above signal line in 12h uptrend, short when crosses below in downtrend.
# The triple smoothing reduces noise while maintaining responsiveness to trend changes.
# Cooldown of 2 bars limits trades to ~30-80 per year. Position size 0.25 manages risk.
# Works in bull markets (captures uptrend continuations) and bear markets (captures downtrends).