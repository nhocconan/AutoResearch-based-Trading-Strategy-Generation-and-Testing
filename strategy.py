#!/usr/bin/env python3
"""
12h_1w_Donchian_Trend_v1
Hypothesis: Trade Donchian channel breakouts on 12h timeframe with weekly trend filter (price above/below weekly EMA20) and volume confirmation (volume > 1.5x 20-period average). 
Designed for low-frequency, high-conviction trades that work in both bull (breakouts continue) and bear (breakouts fail, reverse) markets. 
Targets 15-30 trades per year to minimize fee drag while capturing significant trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_Donchian_Trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === WEEKLY DATA FOR TREND FILTER ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Weekly EMA20 for trend filter
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # === 12H INDICATORS: DONCHIAN CHANNEL AND VOLUME ===
    # Donchian Channel (20-period)
    donchian_period = 20
    donchian_high = np.full_like(high, np.nan)
    donchian_low = np.full_like(low, np.nan)
    
    for i in range(donchian_period - 1, len(high)):
        donchian_high[i] = np.max(high[i - donchian_period + 1:i + 1])
        donchian_low[i] = np.min(low[i - donchian_period + 1:i + 1])
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        # Skip if not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema20_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: price relative to weekly EMA20
        uptrend = close[i] > ema20_1w_aligned[i]
        downtrend = close[i] < ema20_1w_aligned[i]
        
        # Volume confirmation
        strong_volume = volume[i] > (vol_ma[i] * 1.5)
        
        # Long: price breaks above Donchian high in uptrend with volume
        long_signal = (close[i] > donchian_high[i] and 
                      uptrend and 
                      strong_volume)
        
        # Short: price breaks below Donchian low in downtrend with volume
        short_signal = (close[i] < donchian_low[i] and 
                       downtrend and 
                       strong_volume)
        
        # Exit: price returns to opposite Donchian band or trend reverses
        exit_long = (position == 1 and 
                    (close[i] < donchian_low[i] or not uptrend))
        exit_short = (position == -1 and 
                     (close[i] > donchian_high[i] or not downtrend))
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals