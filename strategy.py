#!/usr/bin/env python3
# 4h_12h_adx_trend_v1
# Strategy: 4h trend following with ADX filter and 12h EMA trend confirmation
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: ADX > 25 indicates strong trend; 12h EMA confirms higher timeframe direction.
# Long when ADX > 25, price > 12h EMA, and price > 20-period high; short when ADX > 25, price < 12h EMA, and price < 20-period low.
# Designed for low frequency (15-25 trades/year) to minimize fee drag in trending markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_adx_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # ADX calculation
    period = 14
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    tr = np.maximum(high[1:] - low[1:], np.absolute(high[1:] - close[:-1]), np.absolute(low[1:] - close[:-1]))
    plus_dm = np.insert(plus_dm, 0, 0)
    minus_dm = np.insert(minus_dm, 0, 0)
    tr = np.insert(tr, 0, 0)
    
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean()
    plus_di = 100 * (pd.Series(plus_dm).rolling(window=period, min_periods=period).sum() / atr)
    minus_di = 100 * (pd.Series(minus_dm).rolling(window=period, min_periods=period).sum() / atr)
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = pd.Series(dx).rolling(window=period, min_periods=period).mean()
    
    # Price channels for entry
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min()
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(adx.iloc[i]) or 
            np.isnan(high_20.iloc[i]) or np.isnan(low_20.iloc[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # ADX trend strength filter
        strong_trend = adx.iloc[i] > 25
        
        # Entry conditions
        if strong_trend and close[i] > ema_50_12h_aligned[i] and close[i] > high_20.iloc[i-1] and position != 1:
            position = 1
            signals[i] = 0.25
        elif strong_trend and close[i] < ema_50_12h_aligned[i] and close[i] < low_20.iloc[i-1] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: trend weakening or opposite signal
        elif position == 1 and (adx.iloc[i] < 20 or close[i] < ema_50_12h_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (adx.iloc[i] < 20 or close[i] > ema_50_12h_aligned[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals