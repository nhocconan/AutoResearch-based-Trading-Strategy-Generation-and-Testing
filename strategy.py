#!/usr/bin/env python3
# 4h_1d_volatility_breakout_v1
# Strategy: 4-hour volatility breakout with 1-day trend filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Volatility breakouts combined with daily trend filtering capture
# institutional momentum while filtering false signals. Uses ATR-based volatility
# expansion and daily EMA trend filter. Works in bull markets by catching
# continuation breakouts and in bear markets by capturing breakdowns with
# volatility confirmation. Targets 20-40 trades/year to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_volatility_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily EMA(34) for trend filter (faster than 50 for more signals)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 4h ATR(14) for volatility measurement
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift(1))
    tr3 = abs(low_series - close_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=14).mean().values
    
    # 4h ATR ratio: current ATR / 50-period average ATR (volatility expansion)
    atr_ratio = atr_14 / (pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values + 1e-10)
    
    # 4h Donchian channel (20-period) for breakout levels
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after ATR and Donchian warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr_ratio[i]) or np.isnan(ema_34_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volatility expansion: ATR ratio > 1.5 (current volatility 50% above average)
        vol_expansion = atr_ratio[i] > 1.5
        
        # Breakout conditions
        bull_breakout = close[i] > donchian_high[i-1]  # Break above prior high
        bear_breakout = close[i] < donchian_low[i-1]   # Break below prior low
        
        # Trend filter: price above/below daily EMA34
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Entry logic: volatility expansion + breakout + trend alignment
        if vol_expansion and bull_breakout and uptrend and position != 1:
            position = 1
            signals[i] = 0.25
        elif vol_expansion and bear_breakout and downtrend and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: opposite breakout with volatility confirmation
        elif position == 1 and bear_breakout and vol_expansion:
            position = 0
            signals[i] = 0.0
        elif position == -1 and bull_breakout and vol_expansion:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals