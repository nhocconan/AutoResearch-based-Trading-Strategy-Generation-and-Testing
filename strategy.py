#!/usr/bin/env python3
"""
1d_1w_adx_momentum_v1
Strategy: 1d ADX momentum with 1w trend filter
Timeframe: 1d
Leverage: 1.0
Hypothesis: Uses ADX(14) > 25 to identify strong trends on daily, combined with 1-week EMA20 trend filter for direction. Enters long when both ADX>25 and price>weekly EMA20, short when ADX>25 and price<weekly EMA20. Exits when ADX<20 (trend weakening). Designed to capture strong trends in both bull and bear markets while avoiding choppy periods. Target: 20-50 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_adx_momentum_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load higher timeframe data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 1d ADX(14) for trend strength
    # Calculate True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Directional Movement
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Avoid division by zero
    atr_safe = np.where(atr == 0, 1e-10, atr)
    
    # Smoothed values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_safe
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_safe
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # 1-week EMA20 for trend direction
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(adx[i]) or np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Trend strength filter
        strong_trend = adx[i] > 25
        weak_trend = adx[i] < 20
        
        # Direction from weekly EMA
        price_above_weekly_ema = price_close > ema_20_1w_aligned[i]
        price_below_weekly_ema = price_close < ema_20_1w_aligned[i]
        
        # Entry conditions
        long_entry = strong_trend and price_above_weekly_ema
        short_entry = strong_trend and price_below_weekly_ema
        
        # Exit conditions
        exit_long = position == 1 and (weak_trend or price_below_weekly_ema)
        exit_short = position == -1 and (weak_trend or price_above_weekly_ema)
        
        # Trading logic
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
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

# Hypothesis: Uses ADX(14) > 25 to identify strong trends on daily, combined with 1-week EMA20 trend filter for direction. Enters long when both ADX>25 and price>weekly EMA20, short when ADX>25 and price<weekly EMA20. Exits when ADX<20 (trend weakening). Designed to capture strong trends in both bull and bear markets while avoiding choppy periods. Target: 20-50 trades over 4 years.