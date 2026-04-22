#!/usr/bin/env python3
"""
Hypothesis: 12-hour Donchian(20) breakout with 1-day EMA50 trend and volume confirmation.
Long when price breaks above 12h Donchian upper band with 1-day EMA50 rising and volume spike.
Short when price breaks below 12h Donchian lower band with 1-day EMA50 falling and volume spike.
Exit when price crosses back below/above the 12h Donchian midpoint.
Donchian channels provide trend-following structure; 1-day EMA50 filters for daily trend;
volume spike confirms momentum. Designed for low trade frequency by requiring multiple
confirmations and using 12h-level structure. Works in both bull and bear markets by
following the daily trend and using volatility-based exits.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-day data for EMA50 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1-day EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 12h Donchian channels (20-period high/low)
    # We'll compute these on the 12h price data using rolling window
    # Since we're working with 12h timeframe, we can use the prices directly
    high_12h = high
    low_12h = low
    close_12h = close
    
    # Donchian upper band: 20-period high
    donch_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    # Donchian lower band: 20-period low
    donch_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    # Donchian midpoint: (upper + lower) / 2
    donch_mid = (donch_high + donch_low) / 2.0
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after enough data for EMA50 and Donchian
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(donch_mid[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: Price breaks above Donchian upper band with 1-day EMA50 rising and volume spike
            if (close[i] > donch_high[i] and 
                ema50_1d_aligned[i] > ema50_1d_aligned[i-1] and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower band with 1-day EMA50 falling and volume spike
            elif (close[i] < donch_low[i] and 
                  ema50_1d_aligned[i] < ema50_1d_aligned[i-1] and vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses back below/above Donchian midpoint
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below Donchian midpoint
                if close[i] < donch_mid[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses above Donchian midpoint
                if close[i] > donch_mid[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian_20_1dEMA50_Trend_Volume"
timeframe = "12h"
leverage = 1.0