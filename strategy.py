#!/usr/bin/env python3
"""
6h_RSI2_CloseReversal_1dTrend_HTF
Hypothesis: Extreme short-term RSI(2) reversals on 6h timeframe, filtered by 1d EMA50 trend direction. 
In bull markets (price > EMA50), look for RSI(2) < 10 for long entries. 
In bear markets (price < EMA50), look for RSI(2) > 90 for short entries. 
Exit on RSI(2) crossing back to neutral (40 for longs, 60 for shorts) or opposite extreme.
Designed for low trade frequency (12-37/year) with discrete sizing to minimize fee drag.
Works in both bull and bear markets by only trading with the 1d trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Calculate RSI(2) - very short term for mean reversion signals
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    # Wilder's smoothing (equivalent to EMA with alpha=1/period)
    avg_gain = pd.Series(gain).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi2 = 100 - (100 / (1 + rs))
    
    # Get 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need enough for RSI calculation and EMA
    start_idx = max(10, 50)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(rsi2[i]) or np.isnan(ema_50_aligned[i]):
            signals[i] = 0.0
            continue
        
        rsi_val = rsi2[i]
        ema_trend = ema_50_aligned[i]
        close_val = close[i]
        
        if position == 0:
            # Flat - look for entry
            # Long: RSI(2) < 10 (extremely oversold) AND price above 1d EMA50 (bullish regime)
            # Short: RSI(2) > 90 (extremely overbought) AND price below 1d EMA50 (bearish regime)
            if rsi_val < 10 and close_val > ema_trend:
                signals[i] = size
                position = 1
            elif rsi_val > 90 and close_val < ema_trend:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Long - exit when RSI(2) crosses back above 40 (recovery) or goes extremely overbought
            if rsi_val > 40 or rsi_val > 90:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when RSI(2) crosses back below 60 (recovery) or goes extremely oversold
            if rsi_val < 60 or rsi_val < 10:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_RSI2_CloseReversal_1dTrend_HTF"
timeframe = "6h"
leverage = 1.0