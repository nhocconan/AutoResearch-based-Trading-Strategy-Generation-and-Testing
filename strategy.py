#!/usr/bin/env python3
"""
12h_1d_Donchian_Breakout_TrendFilter_Volume
Hypothesis: Uses daily Donchian channel breakout for entries, filtered by 1-day EMA trend and volume spikes.
Designed for low trade frequency (15-25/year) on 12h timeframe by requiring strong breakouts with volume confirmation.
Works in bull markets via long breakouts and bear markets via short breakouts, with trend filter reducing false signals.
"""

name = "12h_1d_Donchian_Breakout_TrendFilter_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_donchian_channels(high, low, period=20):
    """Calculate Donchian Channels"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    middle = (upper + lower) / 2
    return upper, lower, middle

def calculate_ema(values, period):
    """Calculate Exponential Moving Average"""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Daily Donchian Channel for Breakout Signals ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    donchian_upper, donchian_lower, donchian_middle = calculate_donchian_channels(
        df_1d['high'].values, df_1d['low'].values, period=20
    )
    
    # Align daily Donchian to 12h timeframe
    donchian_upper_12h = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_12h = align_htf_to_ltf(prices, df_1d, donchian_lower)
    donchian_middle_12h = align_htf_to_ltf(prices, df_1d, donchian_middle)
    
    # --- Daily EMA Trend Filter ---
    ema_50 = calculate_ema(df_1d['close'].values, period=50)
    ema_50_12h = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # --- Volume Spike Detection (20-period average on 12h) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_12h[i]) or np.isnan(donchian_lower_12h[i]) or 
            np.isnan(ema_50_12h[i]) or np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation threshold
        volume_spike = vol_ratio[i] > 2.0
        
        if position == 0:
            # Long: price breaks above Donchian upper + above EMA50 + volume spike
            if (close[i] > donchian_upper_12h[i] and 
                close[i] > ema_50_12h[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower + below EMA50 + volume spike
            elif (close[i] < donchian_lower_12h[i] and 
                  close[i] < ema_50_12h[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price returns to Donchian middle
            if position == 1:
                # Exit long: price closes below middle
                if close[i] < donchian_middle_12h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price closes above middle
                if close[i] > donchian_middle_12h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals