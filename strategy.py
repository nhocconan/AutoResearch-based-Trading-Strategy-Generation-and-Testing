#!/usr/bin/env python3
# 6h_Keltner_Channel_Momentum_1dTrend
# Hypothesis: 6-hour price breaking above/below Keltner Channel (2.0 ATR) with 1-day trend filter (EMA50) and volume confirmation.
# Long: price > upper KC + price > EMA50 + volume > 1.5x average.
# Short: price < lower KC + price < EMA50 + volume > 1.5x average.
# Exit: price crosses back to EMA20 (middle KC). Designed for 15-30 trades/year with clear trend following.
# Keltner Channels adapt to volatility, reducing false breakouts in low-vol periods and capturing strong moves.

name = "6h_Keltner_Channel_Momentum_1dTrend"
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate EMA50 for daily trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Keltner Channel components (20-period EMA, 2.0 ATR multiplier)
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr = pd.Series(np.maximum(high - low, np.maximum(abs(high - np.roll(close, 1)), abs(low - np.roll(close, 1))))).rolling(window=20, min_periods=20).mean().values
    # Handle first ATR value
    atr[0] = high[0] - low[0]
    upper_kc = ema20 + 2.0 * atr
    lower_kc = ema20 - 2.0 * atr
    
    # Volume confirmation: 1.5x average volume (50-period for stability)
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50, 50)  # Ensure we have EMA20, ATR, EMA50, and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(upper_kc[i]) or np.isnan(lower_kc[i]) or 
            np.isnan(ema20[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper KC, price above EMA50 (uptrend), volume spike
            if (close[i] > upper_kc[i] and 
                close[i] > ema50_1d_aligned[i] and 
                volume[i] > 1.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower KC, price below EMA50 (downtrend), volume spike
            elif (close[i] < lower_kc[i] and 
                  close[i] < ema50_1d_aligned[i] and 
                  volume[i] > 1.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses back to or below EMA20 (middle KC)
            if close[i] <= ema20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses back to or above EMA20 (middle KC)
            if close[i] >= ema20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals