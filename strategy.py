#!/usr/bin/env python3
# 12H_1D_1W_TripleTimeframe_Confluence
# Hypothesis: Combine 1w trend filter, 1d momentum confirmation, and 12h price action for high-probability entries.
# Uses 1w EMA50 for long-term trend, 1d RSI for momentum, and 12h Donchian breakout for entry.
# Designed to work in both bull and bear markets by following the higher timeframe trend.
# Target: 12-30 trades/year per symbol (48-120 total over 4 years).

name = "12H_1D_1W_TripleTimeframe_Confluence"
timeframe = "12h"
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
    
    # Get 1d and 1w data
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 20 or len(df_1w) < 10:
        return np.zeros(n)
    
    # 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 1d RSI(14) for momentum
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # 12h Donchian(20) for entry signals
    donchian_window = 20
    highest_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, donchian_window)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(rsi_aligned[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Long-term trend from 1w EMA50
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Momentum from 1d RSI
        rsi_value = rsi_aligned[i]
        rsi_overbought = rsi_value > 70
        rsi_oversold = rsi_value < 30
        
        if position == 0:
            # Enter long: uptrend + RSI not overbought + break above Donchian high
            if uptrend and not rsi_overbought and high[i] > highest_high[i-1]:
                signals[i] = 0.25
                position = 1
            # Enter short: downtrend + RSI not oversold + break below Donchian low
            elif downtrend and not rsi_oversold and low[i] < lowest_low[i-1]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: downtrend or RSI overbought or break below Donchian low
            if downtrend or rsi_overbought or low[i] < lowest_low[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: uptrend or RSI oversold or break above Donchian high
            if uptrend or rsi_oversold or high[i] > highest_high[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals