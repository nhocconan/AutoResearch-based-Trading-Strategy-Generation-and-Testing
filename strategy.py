#!/usr/bin/env python3
# 1h_4h1d_donchian_breakout_volume_v1
# Hypothesis: 1h strategy using 4h Donchian channel breakouts with 1d volume confirmation.
# Long: Price breaks above 4h Donchian upper channel (20-period) with 1d volume > 1.5x 20-day average.
# Short: Price breaks below 4h Donchian lower channel (20-period) with 1d volume > 1.5x 20-day average.
# Exit: Price returns to 4h Donchian middle channel or breaks opposite channel level.
# Uses 4h for trend structure (Donchian channels), 1d for volume filter to avoid false breakouts, 1h for precise entry timing.
# Target: 15-37 trades/year (60-150 total over 4 years) on BTC/ETH/SOL.
# Works in bull markets (captures breakouts) and bear markets (shorts breakdowns) with volume confirmation reducing false signals.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h1d_donchian_breakout_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 4h data for Donchian channels (HTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Upper channel: highest high over 20 periods
    high_4h_series = pd.Series(high_4h)
    donchian_upper = high_4h_series.rolling(window=20, min_periods=20).max().values
    # Lower channel: lowest low over 20 periods
    low_4h_series = pd.Series(low_4h)
    donchian_lower = low_4h_series.rolling(window=20, min_periods=20).min().values
    # Middle channel: average of upper and lower
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    # Align 4h Donchian levels to 1h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_4h, donchian_middle)
    
    # Get 1d data for volume confirmation (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d volume average (20-day)
    volume_1d = df_1d['volume'].values
    volume_1d_series = pd.Series(volume_1d)
    volume_ma_1d = volume_1d_series.rolling(window=20, min_periods=20).mean().values
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(donchian_middle_aligned[i]) or np.isnan(volume_ma_1d_aligned[i]) or
            np.isnan(close[i]) or np.isnan(high[i]) or np.isnan(low[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1h volume > 1.5x 20-day average volume (aligned)
        # Note: Using 1h volume compared to 1d average volume as proxy for volume spike
        volume_confirmed = prices['volume'].iloc[i] > 1.5 * volume_ma_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: Price returns to middle channel or breaks below lower channel
            if close[i] <= donchian_middle_aligned[i] or close[i] < donchian_lower_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: Price returns to middle channel or breaks above upper channel
            if close[i] >= donchian_middle_aligned[i] or close[i] > donchian_upper_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Check for breakout with volume confirmation
            bullish_breakout = (close[i] > donchian_upper_aligned[i]) and volume_confirmed
            bearish_breakout = (close[i] < donchian_lower_aligned[i]) and volume_confirmed
            
            if bullish_breakout:
                position = 1
                signals[i] = 0.20
            elif bearish_breakout:
                position = -1
                signals[i] = -0.20
    
    return signals