#!/usr/bin/env python3
# 1h_4h_1d_fvg_confluence_v1
# Strategy: Fair Value Gap (FVG) confluence with 4h/1d trend and volume confirmation
# Timeframe: 1h
# Leverage: 1.0
# Hypothesis: FVGs represent institutional order flow imbalances. In bull markets, long at bullish FVG retests with 4h/1d uptrend and volume. In bear markets, short at bearish FVG retests with 4h/1d downtrend and volume. Uses higher timeframes for direction and 1h for precise entry, targeting 15-35 trades/year to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_fvg_confluence_v1"
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
    volume = prices['volume'].values
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h EMA(50) for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    # Session filter: 08:00-20:00 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Detect FVGs on 1h data
    bullish_fvg = np.zeros(n, dtype=bool)
    bearish_fvg = np.zeros(n, dtype=bool)
    
    for i in range(2, n):
        # Bullish FVG: gap between low[i-2] and high[i]
        if low[i] > high[i-2]:
            bullish_fvg[i] = True
        # Bearish FVG: gap between high[i-2] and low[i]
        if high[i-2] < low[i]:
            bearish_fvg[i] = True
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            not session_filter[i]):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Trend alignment: both 4h and 1d must agree
        uptrend_4h = close[i] > ema_50_4h_aligned[i]
        uptrend_1d = close[i] > ema_50_1d_aligned[i]
        downtrend_4h = close[i] < ema_50_4h_aligned[i]
        downtrend_1d = close[i] < ema_50_1d_aligned[i]
        
        uptrend = uptrend_4h and uptrend_1d
        downtrend = downtrend_4h and downtrend_1d
        
        # Entry logic: FVG retest with volume and trend alignment
        if bullish_fvg[i] and vol_confirm[i] and uptrend and position != 1:
            position = 1
            signals[i] = 0.20
        elif bearish_fvg[i] and vol_confirm[i] and downtrend and position != -1:
            position = -1
            signals[i] = -0.20
        # Exit: opposite FVG forms or trend breaks
        elif position == 1 and (bearish_fvg[i] or not uptrend):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (bullish_fvg[i] or not downtrend):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals