#!/usr/bin/env python3
"""
1d_1w_donchian_breakout_v1
Strategy: 1d Donchian(20) breakout with 1w trend filter and volume confirmation
Timeframe: 1d
Leverage: 1.0
Hypothesis: Uses Donchian channel breakouts on daily timeframe with weekly trend filter (price above/below weekly EMA20) and volume > 1.5x average to avoid false breakouts. Designed to capture strong trends in both bull and bear markets while minimizing false signals. Target: 20-50 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_donchian_breakout_v1"
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
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate weekly EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate average volume (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        vol_ratio = volume[i] / avg_volume[i] if avg_volume[i] > 0 else 0
        
        # Trend filters
        uptrend_1w = price_close > ema_20_1w_aligned[i]
        downtrend_1w = price_close < ema_20_1w_aligned[i]
        
        # Breakout conditions with volume confirmation
        breakout_up = price_close > donchian_high[i-1] and vol_ratio > 1.5
        breakout_down = price_close < donchian_low[i-1] and vol_ratio > 1.5
        
        # Exit when price returns to middle of Donchian channel
        donchian_mid = (donchian_high[i] + donchian_low[i]) / 2
        exit_long = position == 1 and price_close < donchian_mid
        exit_short = position == -1 and price_close > donchian_mid
        
        # Trading logic
        if breakout_up and uptrend_1w and position != 1:
            position = 1
            signals[i] = 0.25
        elif breakout_down and downtrend_1w and position != -1:
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

# Hypothesis: Uses Donchian channel breakouts on daily timeframe with weekly trend filter (price above/below weekly EMA20) and volume > 1.5x average to avoid false breakouts. Designed to capture strong trends in both bull and bear markets while minimizing false signals. Target: 20-50 trades over 4 years.