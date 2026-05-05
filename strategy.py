#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using weekly Donchian channel breakout with 1d EMA50 trend filter and volume confirmation
# Long when price breaks above weekly Donchian upper(20) AND price > 1d EMA50 AND volume > 1.5 * avg_volume(20) on 12h
# Short when price breaks below weekly Donchian lower(20) AND price < 1d EMA50 AND volume > 1.5 * avg_volume(20) on 12h
# Exit when price crosses back below/above weekly Donchian midpoint OR volume drops below average
# Uses discrete sizing 0.25 to balance return and risk
# Target: 60-120 total trades over 4 years (15-30/year) for 12h timeframe
# Weekly Donchian provides robust structure from higher timeframe
# 1d EMA50 filters primary trend to avoid counter-trend trades
# Volume confirmation reduces false breakouts
# Works in bull markets (breakouts with uptrend) and bear markets (breakdowns with downtrend)

name = "12h_DonchianW20_1dEMA50_VolumeConfirm"
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
    
    # Get weekly data ONCE before loop for Donchian channel
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:  # Need at least 20 weekly bars for Donchian(20)
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Donchian channel (20-period)
    # Upper = max(high, lookback=20), Lower = min(low, lookback=20)
    high_1w_series = pd.Series(high_1w)
    low_1w_series = pd.Series(low_1w)
    donchian_upper = high_1w_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_1w_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2.0  # Midpoint for exit
    
    # Align weekly Donchian levels to 12h timeframe (wait for completed weekly bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid)
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough for EMA50
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
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
            np.isnan(donchian_mid_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above weekly Donchian upper, above 1d EMA50, volume confirmation, in session
            if close[i] > donchian_upper_aligned[i] and close[i] > ema50_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly Donchian lower, below 1d EMA50, volume confirmation, in session
            elif close[i] < donchian_lower_aligned[i] and close[i] < ema50_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price crosses below weekly Donchian midpoint OR volume drops below average
            if close[i] < donchian_mid_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price crosses above weekly Donchian midpoint OR volume drops below average
            if close[i] > donchian_mid_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals