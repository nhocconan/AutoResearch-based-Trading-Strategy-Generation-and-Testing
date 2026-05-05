#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout with volume confirmation and 1d EMA50 trend filter
# Long when price breaks above 4h Donchian upper (20) AND volume > 1.5 * avg_volume(20) AND price > 1d EMA50
# Short when price breaks below 4h Donchian lower (20) AND volume > 1.5 * avg_volume(20) AND price < 1d EMA50
# Exit when price crosses back through 4h Donchian middle (10) OR volume drops below average
# Uses discrete sizing 0.20 to balance return and risk
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe
# 4h Donchian provides structure and breakout signals
# 1d EMA50 filters for primary trend alignment to avoid counter-trend trades
# Volume spike confirms breakout strength and reduces false signals
# Works in bull markets (buying breakouts in uptrend) and bear markets (selling breakdowns in downtrend)
# Session filter (08-20 UTC) reduces noise trades during low-liquidity periods

name = "1h_Donchian20_1dEMA50_VolumeSpike"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data ONCE before loop for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:  # Need at least one completed 4h bar for Donchian
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h Donchian channels (20-period)
    highest_high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_upper = highest_high_20
    donchian_lower = lowest_low_20
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    # Align 4h Donchian to 1h timeframe (wait for completed 4h bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_4h, donchian_middle)
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough for EMA50
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 4h Donchian upper, volume confirmation, above 1d EMA50, in session
            if (close[i] > donchian_upper_aligned[i] and close[i-1] <= donchian_upper_aligned[i-1] and 
                volume_confirm[i] and close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h Donchian lower, volume confirmation, below 1d EMA50, in session
            elif (close[i] < donchian_lower_aligned[i] and close[i-1] >= donchian_lower_aligned[i-1] and 
                  volume_confirm[i] and close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price crosses back below 4h Donchian middle OR volume drops below average
            if close[i] < donchian_middle_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price crosses back above 4h Donchian middle OR volume drops below average
            if close[i] > donchian_middle_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals