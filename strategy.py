#!/usr/bin/env python3
"""
1d_1w_floodgate_strategy_v1
Strategy: Daily 20-period Donchian breakout with weekly trend filter and volume confirmation
Timeframe: 1d
Leverage: 1.0
Hypothesis: Uses daily Donchian channel (20-day high/low) for breakout entries, filtered by weekly EMA50 trend direction and confirmed by volume > 1.5x 20-day average. Designed to capture major trend continuations in both bull and bear markets while avoiding false breakouts in choppy conditions. Weekly trend filter ensures alignment with higher timeframe momentum. Target: 30-80 trades over 4 years (7-20/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_floodgate_strategy_v1"
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
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 20-period Donchian channel (daily)
    # Upper band = 20-day high, Lower band = 20-day low
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Trend filter: price above/below weekly EMA50
        uptrend_1w = price_close > ema_50_1w_aligned[i]
        downtrend_1w = price_close < ema_50_1w_aligned[i]
        
        # Breakout conditions using Donchian channels
        breakout_up = price_close > donchian_high[i]
        breakout_down = price_close < donchian_low[i]
        
        # Volume confirmation
        vol_confirmed = vol_spike[i]
        
        # Long: upward breakout with volume in uptrend (weekly)
        long_signal = breakout_up and vol_confirmed and uptrend_1w
        
        # Short: downward breakout with volume in downtrend (weekly)
        short_signal = breakout_down and vol_confirmed and downtrend_1w
        
        # Exit when price returns to the 20-day EMA or opposite Donchian level
        exit_long = position == 1 and (price_close < donchian_low[i])
        exit_short = position == -1 and (price_close > donchian_high[i])
        
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