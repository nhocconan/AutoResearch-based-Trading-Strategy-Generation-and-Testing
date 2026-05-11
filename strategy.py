#!/usr/bin/env python3
"""
4h_1d_RSI_Pullback_to_EMA_Trend
Hypothesis: In 4h, buy pullbacks to EMA50 during 1d uptrend when RSI(14) < 30 (oversold),
sell rallies to EMA50 during 1d downtrend when RSI(14) > 70 (overbought).
Exit when price crosses EMA50 or RSI reverts to neutral.
Uses 1d EMA50 trend filter to avoid counter-trend trades.
Designed for low trade frequency (~20-30/year) to minimize fee drag.
Works in bull (buy dips) and bear (sell rallies) markets.
"""

name = "4h_1d_RSI_Pullback_to_EMA_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 4h OHLCV
    close_4h = prices['close'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    
    # --- 1d Trend Filter: EMA50 ---
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # --- 4h RSI(14) ---
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # --- 4h EMA50 for entry/exit ---
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50  # for EMA50 and RSI
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(rsi[i]) or
            np.isnan(ema50_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1d trend
        trend_up = close_4h[i] > ema50_1d_aligned[i]
        trend_down = close_4h[i] < ema50_1d_aligned[i]
        
        if position == 0:
            # Look for entries only in direction of 1d trend
            if trend_up and rsi[i] < 30 and close_4h[i] <= ema50_4h[i]:
                # Long: 1d uptrend, RSI oversold, price at or below EMA50
                signals[i] = 0.25
                position = 1
            elif trend_down and rsi[i] > 70 and close_4h[i] >= ema50_4h[i]:
                # Short: 1d downtrend, RSI overbought, price at or above EMA50
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price crosses above EMA50 or RSI returns to neutral
                if close_4h[i] > ema50_4h[i] or rsi[i] > 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price crosses below EMA50 or RSI returns to neutral
                if close_4h[i] < ema50_4h[i] or rsi[i] < 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals