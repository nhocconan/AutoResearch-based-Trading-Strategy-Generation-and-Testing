#!/usr/bin/env python3
# 12h_1w_ema_breakout_v1
# Strategy: 12h EMA breakout with weekly trend filter and volume confirmation
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Price breaking above 12h EMA(50) signals bullish momentum, below signals bearish.
# Weekly EMA(50) trend filter ensures trades align with higher timeframe momentum.
# Volume spike filter confirms breakout strength. Works in bull by riding uptrends,
# and in bear by capturing short-term reversals against the weekly trend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_ema_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1w_trend = close_1w > ema_1w  # Uptrend when price > EMA
    ema_1w_trend_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_trend)
    
    # 12h EMA(50) for entry signal
    ema_12h = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume average (20-period) for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)  # Volume spike filter
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_12h[i]) or np.isnan(ema_1w_trend_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Entry conditions
        long_signal = (close[i] > ema_12h[i]) and ema_1w_trend_aligned[i] and vol_spike[i]
        short_signal = (close[i] < ema_12h[i]) and (not ema_1w_trend_aligned[i]) and vol_spike[i]
        
        # Exit conditions: opposite EMA cross
        exit_long = position == 1 and close[i] < ema_12h[i]
        exit_short = position == -1 and close[i] > ema_12h[i]
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals