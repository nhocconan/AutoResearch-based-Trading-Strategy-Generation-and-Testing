#!/usr/bin/env python3
"""
1h_4h1d_Trend_Filtered_Pullback
Hypothesis: In 4h uptrend (close > EMA50) and 1d uptrend (close > EMA200), 
buy pullbacks to EMA20 on 1h with RSI < 40. In 4h downtrend (close < EMA50) 
and 1d downtrend (close < EMA200), sell rallies to EMA20 on 1h with RSI > 60.
Uses session filter (08-20 UTC) to avoid low-volume periods. 
Targets 60-150 total trades over 4 years = 15-37/year for 1h.
"""

name = "1h_4h1d_Trend_Filtered_Pullback"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Precompute hour for session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # 1h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1h EMA20 for pullback entries
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # 1h RSI(14) for overbought/oversold
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # 4h EMA50 for trend
    ema50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 1d EMA200 for trend
    ema200_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 200
    
    for i in range(start_idx, n):
        # Session filter: only trade 08-20 UTC
        if hours[i] < 8 or hours[i] > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any values are NaN
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(ema20[i]) or np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend alignment
        uptrend_4h = close[i] > ema50_4h_aligned[i]
        uptrend_1d = close[i] > ema200_1d_aligned[i]
        downtrend_4h = close[i] < ema50_4h_aligned[i]
        downtrend_1d = close[i] < ema200_1d_aligned[i]
        
        if position == 0:
            # Look for long setup: uptrend on both timeframes + pullback to EMA20 + RSI < 40
            if uptrend_4h and uptrend_1d:
                if close[i] <= ema20[i] and rsi[i] < 40:
                    signals[i] = 0.20
                    position = 1
            # Look for short setup: downtrend on both timeframes + rally to EMA20 + RSI > 60
            elif downtrend_4h and downtrend_1d:
                if close[i] >= ema20[i] and rsi[i] > 60:
                    signals[i] = -0.20
                    position = -1
        else:
            # Manage existing position
            if position == 1:
                # Exit long: trend breakdown or RSI > 70
                if not (uptrend_4h and uptrend_1d) or rsi[i] > 70:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            elif position == -1:
                # Exit short: trend reversal or RSI < 30
                if not (downtrend_4h and downtrend_1d) or rsi[i] < 30:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals