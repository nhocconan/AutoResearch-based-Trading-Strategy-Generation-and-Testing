#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Donchian(20) breakout with 1w Supertrend filter and volume confirmation
# Long when price breaks above 1d Donchian upper(20) AND 1w Supertrend = uptrend AND volume > 1.5 * avg_volume(20) on 12h
# Short when price breaks below 1d Donchian lower(20) AND 1w Supertrend = downtrend AND volume > 1.5 * avg_volume(20) on 12h
# Exit when price crosses back through the 1d Donchian midpoint (upper+lower)/2
# Uses discrete sizing 0.25 to balance return and risk
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# 1d Donchian(20) provides strong breakout levels that reduce whipsaw
# 1w Supertrend filter ensures we trade with the dominant weekly trend
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
    
    # Calculate 1d Donchian(20) levels
    # Donchian upper = max(high, lookback=20), Donchian lower = min(low, lookback=20)
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    donchian_upper_1d = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower_1d = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid_1d = (donchian_upper_1d + donchian_lower_1d) / 2.0
    
    # Align 1d Donchian to 12h timeframe (wait for completed 1d bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper_1d)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower_1d)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid_1d)
    
    # Get 1w data ONCE before loop for Supertrend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:  # Need at least 10 completed weekly bars for Supertrend(10,3)
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Supertrend(10,3)
    # ATR(10)
    tr1 = np.maximum(high_1w, np.roll(close_1w, 1)) - np.minimum(low_1w, np.roll(close_1w, 1))
    tr1[0] = high_1w[0] - low_1w[0]  # First TR
    atr = pd.Series(tr1).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Basic Upper and Lower Bands
    basic_ub = (high_1w + low_1w) / 2.0 + 3.0 * atr
    basic_lb = (high_1w + low_1w) / 2.0 - 3.0 * atr
    
    # Final Upper and Lower Bands
    final_ub = np.zeros_like(close_1w)
    final_lb = np.zeros_like(close_1w)
    supertrend = np.zeros_like(close_1w)
    trend = np.ones_like(close_1w)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close_1w)):
        # Final Upper Band
        if basic_ub[i] < final_ub[i-1] or close_1w[i-1] > final_ub[i-1]:
            final_ub[i] = basic_ub[i]
        else:
            final_ub[i] = final_ub[i-1]
        
        # Final Lower Band
        if basic_lb[i] > final_lb[i-1] or close_1w[i-1] < final_lb[i-1]:
            final_lb[i] = basic_lb[i]
        else:
            final_lb[i] = final_lb[i-1]
        
        # Supertrend and Trend
        if trend[i-1] == -1 and close_1w[i] > final_ub[i]:
            trend[i] = 1
        elif trend[i-1] == 1 and close_1w[i] < final_lb[i]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]
        
        supertrend[i] = final_ub[i] if trend[i] == -1 else final_lb[i]
    
    # Initialize first values
    if len(close_1w) > 0:
        supertrend[0] = final_lb[0]
        trend[0] = 1
    
    # Align 1w Supertrend trend to 12h timeframe (wait for completed 1w bar)
    trend_aligned = align_htf_to_ltf(prices, df_1w, trend.astype(float))
    
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
            np.isnan(trend_aligned[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1d Donchian upper, 1w Supertrend uptrend (trend=1), volume confirmation, in session
            if (close[i] > donchian_upper_aligned[i] and 
                trend_aligned[i] == 1 and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Donchian lower, 1w Supertrend downtrend (trend=-1), volume confirmation, in session
            elif (close[i] < donchian_lower_aligned[i] and 
                  trend_aligned[i] == -1 and 
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