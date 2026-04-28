#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1w Donchian channel breakout with 1d EMA trend filter and volume confirmation.
# Enter long when price breaks above 1w Donchian upper channel (20-period), price > 1d EMA50, and volume > 2.0x 20-bar average.
# Enter short when price breaks below 1w Donchian lower channel, price < 1d EMA50, and volume > 2.0x 20-bar average.
# Exit when price crosses the 1w Donchian midline (average of upper/lower channel) or opposite breakout occurs.
# Donchian provides clear structure, EMA50 filters for trend alignment, volume confirms breakout strength.
# Works in bull markets (breakouts with trend) and bear markets (breakouts against trend still valid with volume).
# Uses discrete position sizing (0.25) to control risk. Target: 50-150 total trades over 4 years.

name = "12h_Donchian20_1wEMA50_VolumeBreakout_v1"
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
    
    # Get 1w data for Donchian channel (HTF)
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Get 1d data for EMA50 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1w Donchian channel (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    donchian_upper = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 12h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate 12h volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(ema_50_aligned[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        breakout_up = close[i] > donchian_upper_aligned[i]
        breakout_down = close[i] < donchian_lower_aligned[i]
        
        # Trend filter conditions
        price_above_ema = close[i] > ema_50_aligned[i]
        price_below_ema = close[i] < ema_50_aligned[i]
        
        # Entry conditions
        long_entry = breakout_up and price_above_ema and volume_confirm[i]
        short_entry = breakout_down and price_below_ema and volume_confirm[i]
        
        # Exit conditions: price crosses Donchian midline or opposite breakout
        long_exit = close[i] < donchian_mid_aligned[i] or breakout_down
        short_exit = close[i] > donchian_mid_aligned[i] or breakout_up
        
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