#!/usr/bin/env python3
"""
1d_KAMA_RSI_Chop_Filter
Hypothesis: On daily timeframe, use KAMA (adaptive trend) for direction, RSI for entry timing, and Choppiness Index to filter ranging markets. KAMA filters whipsaw, RSI captures mean-reversion entries, and Choppiness Index avoids choppy conditions. Designed for low trade frequency (~10-25/year) to minimize fee decay and work in both bull and bear markets.
"""

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
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 34:
        return np.zeros(n)
    
    # Calculate weekly EMA34 for trend filter
    close_weekly = df_weekly['close'].values
    ema34_weekly = pd.Series(close_weekly).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly EMA34 to daily timeframe
    ema34_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema34_weekly)
    
    # KAMA on daily close (ER=10)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close)).cumsum() - np.abs(np.diff(close, prepend=close[0])).cumsum()
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14) on daily close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14) on daily
    atr = np.zeros(n)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[high[0] - low[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_atr / (highest_high - lowest_low)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_weekly_aligned[i]) or np.isnan(kama[i]) or
            np.isnan(rsi[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Conditions
        kama_up = close[i] > kama[i]
        kama_down = close[i] < kama[i]
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        chop_high = chop[i] > 61.8  # ranging market
        
        # Entry: KAMA direction + RSI extreme + not choppy
        long_entry = kama_up and rsi_oversold and not chop_high
        short_entry = kama_down and rsi_overbought and not chop_high
        
        # Exit: opposite RSI extreme or KAMA flip
        long_exit = (not kama_up) or rsi[i] > 50
        short_exit = (not kama_down) or rsi[i] < 50
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_KAMA_RSI_Chop_Filter"
timeframe = "1d"
leverage = 1.0