#!/usr/bin/env python3
# Hypothesis: 1d Weekly Donchian (20) breakout with 1w EMA(50) trend filter and volume confirmation.
# Uses weekly Donchian channels from 1w to identify key support/resistance. Breakouts above upper band or below lower band
# trigger entries in the direction of the 1w EMA(50) trend. Volume confirmation (1.5x 20-period average)
# filters false breakouts. Designed for 1d timeframe with ~30-100 total trades over 4 years to minimize fee drift.
# Works in both bull and bear markets due to trend filter and volume confirmation.

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
    
    # Get 1w data for Donchian channels and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian channels from previous 1w bar (to avoid look-ahead)
    # Upper band = max(high, 20), Lower band = min(low, 20) from previous week
    prev_high = df_1w['high'].shift(1).values
    prev_low = df_1w['low'].shift(1).values
    
    # Calculate 20-period Donchian channels
    high_series = pd.Series(prev_high)
    low_series = pd.Series(prev_low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d timeframe (they are constant for the week)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower)
    
    # Volume filter: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Wait for EMA and volume
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1w EMA(50)
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Entry conditions: breakout from Donchian bands in trend direction with volume
        long_breakout = close[i] > donchian_upper_aligned[i]
        short_breakout = close[i] < donchian_lower_aligned[i]
        
        long_entry = long_breakout and uptrend and volume_confirm[i]
        short_entry = short_breakout and downtrend and volume_confirm[i]
        
        # Exit conditions: price returns to opposite Donchian band or trend reversal
        long_exit = (close[i] < donchian_lower_aligned[i]) or (not uptrend)
        short_exit = (close[i] > donchian_upper_aligned[i]) or (not downtrend)
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_Weekly20Breakout_1wEMA50_Trend_Volume"
timeframe = "1d"
leverage = 1.0