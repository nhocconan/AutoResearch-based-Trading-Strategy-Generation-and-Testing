#!/usr/bin/env python3
"""
6h_1w_volatility_breakout_v1
Strategy: 6h volatility breakout with weekly filter
Timeframe: 6h
Leverage: 1.0
Hypothesis: Uses ATR-based volatility breakout on 6h combined with weekly trend filter to capture strong directional moves while avoiding false breakouts in chop. Works in both bull and bear markets by following the higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_volatility_breakout_v1"
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
    
    # Load higher timeframe data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 6h ATR(14) for volatility breakout
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 6h Donchian channel breakout levels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(atr[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Trend filter
        uptrend = price_close > ema_50_1w_aligned[i]
        downtrend = price_close < ema_50_1w_aligned[i]
        
        # Volatility breakout conditions
        volatility_expansion = atr[i] > 1.5 * pd.Series(atr).rolling(window=30, min_periods=30).mean().values[i] if i >= 30 else False
        breakout_up = price_close > donchian_high[i-1]  # Break above previous period high
        breakout_down = price_close < donchian_low[i-1]  # Break below previous period low
        
        # Long: Breakout up in uptrend with volatility expansion
        long_signal = breakout_up and uptrend and volatility_expansion
        
        # Short: Breakout down in downtrend with volatility expansion
        short_signal = breakout_down and downtrend and volatility_expansion
        
        # Exit when price returns to middle of Donchian channel
        donchian_mid = (donchian_high[i] + donchian_low[i]) / 2.0
        exit_long = position == 1 and price_close < donchian_mid
        exit_short = position == -1 and price_close > donchian_mid
        
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

# Hypothesis: Uses ATR-based volatility breakout on 6h combined with weekly trend filter to capture strong directional moves while avoiding false breakouts in chop. Works in both bull and bear markets by following the higher timeframe trend.