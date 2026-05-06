#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Donchian(20) breakout with 1w Supertrend(10,3) trend filter and volume confirmation
# Long when price breaks above 1d Donchian upper(20) AND 1w Supertrend is bullish (below price) AND volume > 1.5 * avg_volume(20) on 12h
# Short when price breaks below 1d Donchian lower(20) AND 1w Supertrend is bearish (above price) AND volume > 1.5 * avg_volume(20) on 12h
# Exit when price crosses back through the 1d Donchian midpoint (upper+lower)/2
# Uses discrete sizing 0.25 to balance return and risk
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# 1d Donchian(20) provides strong breakout levels that reduce whipsaw
# 1w Supertrend(10,3) trend filter ensures we trade with the dominant weekly trend
# Volume confirmation (1.5x) validates breakout strength while limiting overtrading

name = "12h_1dDonchian20_1wSupertrend_Trend_VolumeConfirm"
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
    
    # Get 1d data ONCE before loop for Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need at least 20 completed 1d bars for Donchian(20)
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian channels (20-period)
    # Upper = highest high of last 20 periods, Lower = lowest low of last 20 periods
    donchian_upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_mid_20 = (donchian_upper_20 + donchian_lower_20) / 2.0
    
    # Align 1d Donchian to 12h timeframe (wait for completed 1d bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper_20)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower_20)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid_20)
    
    # Get 1w data ONCE before loop for Supertrend(10,3) trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:  # Need at least 10 completed weekly bars for Supertrend
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Supertrend(10,3)
    # ATR(10)
    tr1 = pd.Series(high_1w - low_1w)
    tr2 = pd.Series(np.abs(high_1w - pd.Series(close_1w).shift(1)))
    tr3 = pd.Series(np.abs(low_1w - pd.Series(close_1w).shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_10 = tr.ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Basic Upper Band = (high + low)/2 + multiplier * ATR
    # Basic Lower Band = (high + low)/2 - multiplier * ATR
    hl2 = (high_1w + low_1w) / 2
    basic_upper = hl2 + 3 * atr_10
    basic_lower = hl2 - 3 * atr_10
    
    # Final Upper Band
    final_upper = np.zeros_like(basic_upper)
    final_upper[0] = basic_upper[0]
    for i in range(1, len(basic_upper)):
        if close_1w[i-1] <= final_upper[i-1]:
            final_upper[i] = min(basic_upper[i], final_upper[i-1])
        else:
            final_upper[i] = basic_upper[i]
    
    # Final Lower Band
    final_lower = np.zeros_like(basic_lower)
    final_lower[0] = basic_lower[0]
    for i in range(1, len(basic_lower)):
        if close_1w[i-1] >= final_lower[i-1]:
            final_lower[i] = max(basic_lower[i], final_lower[i-1])
        else:
            final_lower[i] = basic_lower[i]
    
    # Supertrend: if close <= final_upper then uptrend (1), else downtrend (-1)
    supertrend = np.ones_like(close_1w)
    supertrend[close_1w > final_upper] = -1
    
    # Align 1w Supertrend to 12h timeframe (wait for completed 1w bar)
    supertrend_aligned = align_htf_to_ltf(prices, df_1w, supertrend)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 12h
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
            np.isnan(supertrend_aligned[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1d Donchian upper, 1w Supertrend bullish (1), volume confirmation, in session
            if (close[i] > donchian_upper_aligned[i] and 
                supertrend_aligned[i] == 1 and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Donchian lower, 1w Supertrend bearish (-1), volume confirmation, in session
            elif (close[i] < donchian_lower_aligned[i] and 
                  supertrend_aligned[i] == -1 and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back below 1d Donchian midpoint
            if close[i] < donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back above 1d Donchian midpoint
            if close[i] > donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals