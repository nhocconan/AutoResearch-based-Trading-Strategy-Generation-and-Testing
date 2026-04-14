#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with 1d RSI(14) for momentum and 1w MACD histogram for trend confirmation.
# RSI(14) > 50 indicates bullish momentum on daily timeframe.
# MACD histogram crossing above/below zero on weekly timeframe confirms trend direction.
# Combining these filters reduces whipsaw and captures sustained moves in both bull and bear markets.
# Expected trade frequency: 15-25 per year per symbol, staying within optimal range.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Load 1d data ONCE for RSI
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d RSI(14)
    rsi_length = 14
    delta = np.diff(df_1d['close'].values, prepend=df_1d['close'].values[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/rsi_length, min_periods=rsi_length, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/rsi_length, min_periods=rsi_length, adjust=False).mean().mean().values
    rs = avg_gain / np.where(avg_loss == 0, 1, avg_loss)
    rsi = 100 - (100 / (1 + rs))
    
    # Align RSI to 6h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Load 1w data ONCE for MACD
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w MACD (12,26,9)
    fast_length = 12
    slow_length = 26
    signal_length = 9
    
    close_1w = df_1w['close'].values
    ema_fast = pd.Series(close_1w).ewm(span=fast_length, adjust=False, min_periods=fast_length).mean().values
    ema_slow = pd.Series(close_1w).ewm(span=slow_length, adjust=False, min_periods=slow_length).mean().values
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=signal_length, adjust=False, min_periods=signal_length).mean().values
    macd_hist = macd_line - signal_line
    
    # Align MACD histogram to 6h timeframe
    macd_hist_aligned = align_htf_to_ltf(prices, df_1w, macd_hist)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(30, 26, 9)  # RSI(14) + MACD(26,9)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi_aligned[i]) or 
            np.isnan(macd_hist_aligned[i])):
            signals[i] = 0.0
            continue
        
        rsi_val = rsi_aligned[i]
        macd_hist_val = macd_hist_aligned[i]
        
        if position == 0:
            # Enter long: RSI > 50 (bullish momentum) AND MACD hist > 0 (bullish trend)
            if rsi_val > 50 and macd_hist_val > 0:
                position = 1
                signals[i] = position_size
            # Enter short: RSI < 50 (bearish momentum) AND MACD hist < 0 (bearish trend)
            elif rsi_val < 50 and macd_hist_val < 0:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI < 40 OR MACD hist < 0 (loss of momentum or trend)
            if rsi_val < 40 or macd_hist_val < 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI > 60 OR MACD hist > 0 (loss of bearish momentum or trend)
            if rsi_val > 60 or macd_hist_val > 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1dRSI_1wMACD_Trend_Momentum_v1"
timeframe = "6h"
leverage = 1.0