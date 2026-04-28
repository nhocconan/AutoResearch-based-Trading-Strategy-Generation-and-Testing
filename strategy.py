#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
# Enter long when price breaks above 12h Donchian upper(20) with volume > 2.0x 20-bar average and price > 1d EMA50 (uptrend).
# Enter short when price breaks below 12h Donchian lower(20) with volume > 2.0x 20-bar average and price < 1d EMA50 (downtrend).
# Exit on opposite Donchian level (lower for long exit, upper for short exit) to limit drawdown.
# Uses discrete position sizing (0.30) to control risk. Target: 50-150 total trades over 4 years.
# Donchian provides clear structure, volume confirms breakout strength, 1d EMA50 filters counter-trend noise.
# Works in bull (breakouts with trend) and bear (failed breaks via exits) markets.

name = "12h_Donchian20_1dEMA50_VolumeBreakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian calculation (primary timeframe)
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h Donchian channels (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Donchian upper = max(high, lookback=20)
    # Donchian lower = min(low, lookback=20)
    high_series = pd.Series(high_12h)
    low_series = pd.Series(low_12h)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align 12h Donchian levels to 12h timeframe (same timeframe, so direct use)
    # But we need to ensure we only use completed bars, so align with 0 delay (completed bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower)
    
    # Get 1d data for EMA50 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 12h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 12h volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions with volume confirmation and trend filter
        long_breakout = close[i] > donchian_upper_aligned[i] and volume_confirm[i] and close[i] > ema_50_1d_aligned[i]
        short_breakout = close[i] < donchian_lower_aligned[i] and volume_confirm[i] and close[i] < ema_50_1d_aligned[i]
        
        # Exit conditions: opposite Donchian level
        long_exit = close[i] < donchian_lower_aligned[i]
        short_exit = close[i] > donchian_upper_aligned[i]
        
        # Handle entries and exits
        if long_breakout and position <= 0:
            signals[i] = 0.30
            position = 1
        elif short_breakout and position >= 0:
            signals[i] = -0.30
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.30
            elif position == -1:
                signals[i] = -0.30
            else:
                signals[i] = 0.0
    
    return signals