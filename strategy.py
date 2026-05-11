#!/usr/bin/env python3
"""
12h_RSI_Momentum_Trend_Follow
Hypothesis: In strong trends (weekly EMA50), RSI momentum (3-period) provides timely entries with mean-reversion exits. Weekly trend filter avoids counter-trend trades, while RSI overbought/oversold levels trigger mean-reversion exits. Designed for 12h timeframe to balance trade frequency and capture multi-day moves. Works in both bull and bear markets by following the higher-timeframe trend.
"""

name = "12h_RSI_Momentum_Trend_Follow"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Get daily data for RSI calculation (more responsive)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # 12h OHLCV
    close_12h = prices['close'].values
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    
    # --- Weekly EMA50 for trend filter ---
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # --- Daily RSI(3) for momentum signals ---
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/3, adjust=False, min_periods=3).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/3, adjust=False, min_periods=3).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_12h_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if np.isnan(ema50_1w_aligned[i]) or np.isnan(rsi_12h_aligned[i]):
            if position != 0:
                # Simple stop based on price action
                atr_est = np.abs(high_12h[i] - low_12h[i])  # rough 12h ATR estimate
                if position == 1 and close_12h[i] <= entry_price - 2.0 * atr_est:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_12h[i] >= entry_price + 2.0 * atr_est:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Trend filter: price relative to weekly EMA50
        uptrend = close_12h[i] > ema50_1w_aligned[i]
        downtrend = close_12h[i] < ema50_1w_aligned[i]
        
        # RSI signals
        rsi_value = rsi_12h_aligned[i]
        rsi_overbought = rsi_value > 70
        rsi_oversold = rsi_value < 30
        rsi_bullish = rsi_value > 50
        rsi_bearish = rsi_value < 50
        
        if position == 0:
            # Look for entries in direction of weekly trend
            if uptrend and rsi_bullish and rsi_value < 40:
                # Pullback in uptrend - long entry
                signals[i] = 0.25
                position = 1
                entry_price = close_12h[i]
            elif downtrend and rsi_bearish and rsi_value > 60:
                # Bounce in downtrend - short entry
                signals[i] = -0.25
                position = -1
                entry_price = close_12h[i]
        else:
            # Manage existing position
            if position == 1:
                # Long position: exit on RSI overbought or trend change
                if rsi_overbought or not uptrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short position: exit on RSI oversold or trend change
                if rsi_oversold or not downtrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals