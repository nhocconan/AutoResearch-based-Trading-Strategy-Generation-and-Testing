#!/usr/bin/env python3
# 6h_1d_adx_ema_v1
# Strategy: 6-day ADX/EMA trend following with 1-day EMA filter
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: ADX > 25 identifies strong trends on daily timeframe. EMA(50) on 6h provides entry/exit timing.
# Works in bull markets via ADX + price > EMA50 longs. Works in bear markets via ADX + price < EMA50 shorts.
# Low turnover design targets ~15-30 trades/year to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_adx_ema_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d ADX(14) calculation
    plus_dm = np.diff(df_1d['high'].values, prepend=0)
    minus_dm = np.diff(df_1d['low'].values, prepend=0) * -1
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    tr1 = np.abs(np.diff(df_1d['high'].values, prepend=df_1d['high'].values[0]))
    tr2 = np.abs(np.diff(df_1d['low'].values, prepend=df_1d['low'].values[0]))
    tr3 = np.abs(df_1d['high'].values - df_1d['low'].values)
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / (atr + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 6h EMA50 for entry timing
    ema_50_6h = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if np.isnan(adx_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_50_6h[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # ADX trend strength filter: ADX > 25 indicates strong trend
        strong_trend = adx_aligned[i] > 25
        
        # Entry conditions
        # Long: Strong trend AND price above both 1d and 6h EMA50
        if strong_trend and close[i] > ema_50_1d_aligned[i] and close[i] > ema_50_6h[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Strong trend AND price below both 1d and 6h EMA50
        elif strong_trend and close[i] < ema_50_1d_aligned[i] and close[i] < ema_50_6h[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Trend weakens (ADX < 20) OR price crosses opposite EMA
        elif position == 1 and (adx_aligned[i] < 20 or close[i] < ema_50_6h[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (adx_aligned[i] < 20 or close[i] > ema_50_6h[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals