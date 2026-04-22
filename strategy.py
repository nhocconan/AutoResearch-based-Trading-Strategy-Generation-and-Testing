#!/usr/bin/env python3
"""
Hypothesis: 6-hour RSI with 1-day volume confirmation and 1-week trend filter.
Long when RSI < 30, 1-day volume > 20-period average, and 1-week EMA50 is rising.
Short when RSI > 70, 1-day volume > 20-period average, and 1-week EMA50 is falling.
Exit when RSI returns to neutral (40-60) or volume condition fails.
Combines mean reversion (RSI extremes) with institutional volume confirmation and weekly trend filter to avoid counter-trend trades.
Works in bull markets (buy oversold dips in uptrend) and bear markets (sell overbought rallies in downtrend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI calculation (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Load 1-day data for volume filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    avg_vol_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    
    # Load 1-week data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Determine if weekly EMA50 is rising/falling (using 2-period change)
    ema50_1w_rising = np.zeros(len(ema50_1w_aligned), dtype=bool)
    ema50_1w_falling = np.zeros(len(ema50_1w_aligned), dtype=bool)
    for i in range(2, len(ema50_1w_aligned)):
        ema50_1w_rising[i] = ema50_1w_aligned[i] > ema50_1w_aligned[i-2]
        ema50_1w_falling[i] = ema50_1w_aligned[i] < ema50_1w_aligned[i-2]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if np.isnan(rsi[i]) or np.isnan(avg_vol_1d_aligned[i]) or np.isnan(ema50_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI oversold, volume above average, weekly EMA50 rising
            if rsi[i] < 30 and volume_1d[i] > avg_vol_1d_aligned[i] and ema50_1w_rising[i]:
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought, volume above average, weekly EMA50 falling
            elif rsi[i] > 70 and volume_1d[i] > avg_vol_1d_aligned[i] and ema50_1w_falling[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: RSI returns to neutral (40-60) or volume condition fails
                if rsi[i] >= 40 or volume_1d[i] <= avg_vol_1d_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: RSI returns to neutral (40-60) or volume condition fails
                if rsi[i] <= 60 or volume_1d[i] <= avg_vol_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_RSI_Volume_WeeklyTrend"
timeframe = "6h"
leverage = 1.0