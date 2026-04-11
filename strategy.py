#!/usr/bin/env python3
"""
12h_1w_donchian_breakout_volume_trend_v1
Strategy: 12h Donchian breakout with volume confirmation and 1w trend filter
Timeframe: 12h
Leverage: 1.0
Hypothesis: Uses 12h Donchian channel breakouts for entry, filtered by 1-week EMA20 trend direction and volume spikes (>2x average). Designed to capture major trend continuations in both bull and bear markets with low trade frequency. Uses weekly trend as primary filter to avoid counter-trend trades, volume confirmation to ensure breakout validity, and Donchian channels for objective breakout levels. Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_donchian_breakout_volume_trend_v1"
timeframe = "12h"
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
    
    # 12h Donchian channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1-week EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_avg)  # Volume spike filter
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Trend filter: price above/below 1-week EMA20
        uptrend = price_close > ema_20_1w_aligned[i]
        downtrend = price_close < ema_20_1w_aligned[i]
        
        # Breakout conditions using Donchian channels
        breakout_up = price_close > donchian_high[i]
        breakout_down = price_close < donchian_low[i]
        
        # Volume confirmation
        vol_confirmed = vol_spike[i]
        
        # Long: upward breakout with volume in uptrend
        long_signal = breakout_up and vol_confirmed and uptrend
        
        # Short: downward breakout with volume in downtrend
        short_signal = breakout_down and vol_confirmed and downtrend
        
        # Exit when price returns to the opposite Donchian level
        exit_long = position == 1 and price_close < donchian_low[i]
        exit_short = position == -1 and price_close > donchian_high[i]
        
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