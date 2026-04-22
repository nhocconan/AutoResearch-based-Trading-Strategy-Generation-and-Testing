#!/usr/bin/env python3
"""
Hypothesis: 1-hour trading with 4-hour RSI and 1-day trend filter.
Long when 4-hour RSI < 30 (oversold) and 1-day EMA50 rising, short when 4-hour RSI > 70 (overbought) and 1-day EMA50 falling.
Entries timed on 1-hour chart using price action: buy near support (above EMA20), sell near resistance (below EMA20).
Uses 4-hour for signal generation (low frequency) and 1-hour for precise entry/exit timing.
Designed to work in both bull and bear markets by combining mean reversion on 4h with trend filter on 1d.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 4-hour data for RSI - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    # Calculate RSI(14) on 4h close
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_4h = 100 - (100 / (1 + rs))
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # Load 1-day data for EMA50 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1-hour EMA20 for entry timing
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after enough data for EMA20
        # Skip if data not ready
        if (np.isnan(rsi_4h_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 4h RSI < 30 (oversold) and 1d EMA50 rising, price above 1h EMA20
            if (rsi_4h_aligned[i] < 30 and 
                ema50_1d_aligned[i] > ema50_1d_aligned[i-1] and
                close[i] > ema20[i]):
                signals[i] = 0.20
                position = 1
            # Short: 4h RSI > 70 (overbought) and 1d EMA50 falling, price below 1h EMA20
            elif (rsi_4h_aligned[i] > 70 and 
                  ema50_1d_aligned[i] < ema50_1d_aligned[i-1] and
                  close[i] < ema20[i]):
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: 4h RSI > 50 (momentum fading) OR price falls below EMA20
                if (rsi_4h_aligned[i] > 50 or 
                    close[i] < ema20[i]):
                    exit_signal = True
            else:  # position == -1
                # Exit short: 4h RSI < 50 (momentum fading) OR price rises above EMA20
                if (rsi_4h_aligned[i] < 50 or 
                    close[i] > ema20[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_4hRSI_1dEMA50_Trend_Filter"
timeframe = "1h"
leverage = 1.0