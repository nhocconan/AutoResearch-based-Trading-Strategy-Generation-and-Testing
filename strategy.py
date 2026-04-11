#!/usr/bin/env python3
# 1h_4h_1d_camarilla_breakout_v1
# Strategy: 1h Camarilla breakout with 4h/1d trend filter and session filter
# Timeframe: 1h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels act as strong intraday support/resistance.
# Breakouts above H3 or below L3 with 4h/1d trend alignment capture momentum moves.
# Session filter (08-20 UTC) reduces noise. Designed for 15-30 trades/year to avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_camarilla_breakout_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Warmup for EMA calculations
        # Skip if any required data is invalid or outside session
        if np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or not in_session[i]:
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Calculate Camarilla levels for previous candle
        ph = high[i-1]
        pl = low[i-1]
        pc = close[i-1]
        rang = ph - pl
        
        if rang <= 0:
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Camarilla levels
        h3 = pc + (rang * 1.1 / 6)
        l3 = pc - (rang * 1.1 / 6)
        h4 = pc + (rang * 1.1 / 2)
        l4 = pc - (rang * 1.1 / 2)
        
        # Trend filters
        uptrend_4h = close[i] > ema_50_4h_aligned[i]
        uptrend_1d = close[i] > ema_200_1d_aligned[i]
        downtrend_4h = close[i] < ema_50_4h_aligned[i]
        downtrend_1d = close[i] < ema_200_1d_aligned[i]
        
        # Entry conditions
        # Long: Price breaks above H3 with 4h/1d uptrend
        if close[i] > h3 and uptrend_4h and uptrend_1d and position != 1:
            position = 1
            signals[i] = 0.20
        # Short: Price breaks below L3 with 4h/1d downtrend
        elif close[i] < l3 and downtrend_4h and downtrend_1d and position != -1:
            position = -1
            signals[i] = -0.20
        # Exit: Opposite break of H3/L3
        elif position == 1 and close[i] < l3:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > h3:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals