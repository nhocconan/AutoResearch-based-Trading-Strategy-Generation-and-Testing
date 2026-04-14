#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1-week EMA50 as trend filter with 1-day Donchian(20) breakout
# - Long when weekly trend is up (price above weekly EMA50) and price breaks above daily Donchian high
# - Short when weekly trend is down (price below weekly EMA50) and price breaks below daily Donchian low
# - Volume confirmation: volume > 1.2x 24-period average to ensure participation
# - Uses weekly EMA50 to determine trend direction, works in both bull and bear markets
# - Donchian breakout provides entry timing with clear risk management
# - Target: 50-150 total trades over 4 years (12-37/year) for optimal balance
# - Position size 0.25 for balanced risk exposure

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data once before loop
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50
    close_w = df_w['close'].values
    ema50_w = pd.Series(close_w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate daily Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donch_high = high_series.rolling(window=20, min_periods=20).max().values
    donch_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume filter: 24-period average (2 days of 12h data)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(vol_ma[i]):
            continue
        
        # Get weekly index for current 12h bar
        # 1 week = 7 days = 14 * 12h bars
        idx_w = i // 14
        if idx_w < 1:
            continue
            
        # Previous week's EMA50 (to avoid look-ahead)
        ema50_prev = ema50_w[idx_w-1]
        
        # Create arrays for alignment (constant values for the week)
        ema50_arr = np.full(len(df_w), ema50_prev)
        
        # Align to 12h timeframe
        ema50_12h = align_htf_to_ltf(prices, df_w, ema50_arr)[i]
        
        if position == 0:
            # Long: Weekly trend up + price breaks above Donchian high + volume
            if (close[i] > ema50_12h and  # Weekly trend up
                high[i] > donch_high[i] and  # Break above Donchian high
                volume[i] > vol_ma[i] * 1.2):  # Volume confirmation
                position = 1
                signals[i] = position_size
            # Short: Weekly trend down + price breaks below Donchian low + volume
            elif (close[i] < ema50_12h and  # Weekly trend down
                  low[i] < donch_low[i] and  # Break below Donchian low
                  volume[i] > vol_ma[i] * 1.2):  # Volume confirmation
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: Price breaks below Donchian low or weekly trend turns down
            if low[i] < donch_low[i] or close[i] < ema50_12h:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: Price breaks above Donchian high or weekly trend turns up
            if high[i] > donch_high[i] or close[i] > ema50_12h:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "12h_1w_EMA50_Donchian_Volume"
timeframe = "12h"
leverage = 1.0