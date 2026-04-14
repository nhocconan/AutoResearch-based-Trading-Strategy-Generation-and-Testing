#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly trend as filter with Donchian breakout on daily
# - Long when weekly trend is up (price above weekly EMA20) and price breaks above Donchian(10) high
# - Short when weekly trend is down (price below weekly EMA20) and price breaks below Donchian(10) low
# - Uses weekly EMA20 to determine trend direction, works in both bull and bear markets
# - Donchian breakout provides entry timing with clear risk management
# - Volume confirmation ensures institutional participation at breakout
# - Target: 30-80 total trades over 4 years (7-20/year) to balance opportunity and cost
# - Position size 0.25 for balanced risk exposure

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data once before loop
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA20
    close_w = df_w['close'].values
    ema20_w = pd.Series(close_w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate daily Donchian channels (10-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donch_high = high_series.rolling(window=10, min_periods=10).max().values
    donch_low = low_series.rolling(window=10, min_periods=10).min().values
    
    # Volume filter: 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(30, n):
        # Skip if any critical data is NaN
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(vol_ma[i]):
            continue
        
        # Get weekly index for current daily bar
        idx_w = i // 7  # Approximate weekly index from daily bars
        if idx_w < 1:
            continue
            
        # Previous week's EMA20 (to avoid look-ahead)
        ema20_prev = ema20_w[idx_w-1]
        
        # Create arrays for alignment (constant values for the week)
        ema20_arr = np.full(len(df_w), ema20_prev)
        
        # Align to daily timeframe
        ema20_daily = align_htf_to_ltf(prices, df_w, ema20_arr)[i]
        
        if position == 0:
            # Long: Weekly trend up + price breaks above Donchian high + volume
            if (close[i] > ema20_daily and  # Weekly trend up
                high[i] > donch_high[i] and  # Break above Donchian high
                volume[i] > vol_ma[i] * 1.5):  # Volume confirmation
                position = 1
                signals[i] = position_size
            # Short: Weekly trend down + price breaks below Donchian low + volume
            elif (close[i] < ema20_daily and  # Weekly trend down
                  low[i] < donch_low[i] and  # Break below Donchian low
                  volume[i] > vol_ma[i] * 1.5):  # Volume confirmation
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: Price breaks below Donchian low or weekly trend turns down
            if low[i] < donch_low[i] or close[i] < ema20_daily:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: Price breaks above Donchian high or weekly trend turns up
            if high[i] > donch_high[i] or close[i] > ema20_daily:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "1d_1w_EMA20_Donchian_Volume"
timeframe = "1d"
leverage = 1.0