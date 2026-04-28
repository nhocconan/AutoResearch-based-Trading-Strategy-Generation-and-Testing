#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Donchian breakout with volume confirmation and ATR-based stoploss.
# Enter long when price breaks above 1d Donchian upper channel (20-period high) with volume > 1.5x 20-bar average.
# Enter short when price breaks below 1d Donchian lower channel (20-period low) with volume > 1.5x 20-bar average.
# Exit when price crosses the 10-period EMA of the 1d close (dynamic trailing stop).
# Uses Donchian channels for structure, volume for confirmation, and EMA for trend-following exit.
# Works in bull markets (breakouts with continuation) and bear markets (breakdowns with continuation).
# Discrete position sizing (0.25) to control risk and minimize fee churn. Target: 50-150 total trades over 4 years.

name = "12h_Donchian20_1dEMA10_VolumeBreakout_v1"
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
    
    # Get 1d data for Donchian and EMA calculation (HTF)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian upper channel: 20-period high
    donchian_upper = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Donchian lower channel: 20-period low
    donchian_lower = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    # Donchian middle: 10-period EMA of close for exit
    ema_10 = pd.Series(close_1d).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Align 1d indicators to 12h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    ema_10_aligned = align_htf_to_ltf(prices, df_1d, ema_10)
    
    # Calculate 12h volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema_10_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        bullish_breakout = close[i] > donchian_upper_aligned[i]
        bearish_breakout = close[i] < donchian_lower_aligned[i]
        
        # Exit conditions: price crosses 10-period EMA
        long_exit = close[i] < ema_10_aligned[i]
        short_exit = close[i] > ema_10_aligned[i]
        
        # Entry conditions
        long_entry = bullish_breakout and volume_confirm[i]
        short_entry = bearish_breakout and volume_confirm[i]
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
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