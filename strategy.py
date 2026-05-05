#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using Donchian(20) breakout with 4h EMA50 trend filter and volume confirmation
# Long when price breaks above 4h Donchian upper(20) AND price > 4h EMA50 AND volume > 1.5 * avg_volume(20)
# Short when price breaks below 4h Donchian lower(20) AND price < 4h EMA50 AND volume > 1.5 * avg_volume(20)
# Exit when price crosses back below/above 4h EMA50
# Uses discrete sizing 0.25 to balance return and risk
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Donchian provides robust price channel structure
# 4h EMA50 filters primary trend to avoid counter-trend trades
# Volume confirmation reduces false breakouts
# Works in bull markets (breakouts with uptrend) and bear markets (breakdowns with downtrend)

name = "4h_Donchian20_EMA50_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data ONCE before loop for Donchian and EMA50
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:  # Need enough for EMA50 and Donchian
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate 4h Donchian channels (20-period)
    # Upper = max(high, 20), Lower = min(low, 20)
    high_series = pd.Series(high_4h)
    low_series = pd.Series(low_4h)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h EMA50
    close_series = pd.Series(close_4h)
    ema50_4h = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 4h
    volume_series = pd.Series(volume_4h)
    avg_volume_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume_4h > (1.5 * avg_volume_20)
    
    # Align 4h indicators to 4h timeframe (no additional delay needed for these)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_4h, volume_confirm)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema50_4h_aligned[i]) or np.isnan(volume_confirm_aligned[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above 4h Donchian upper, above 4h EMA50, volume confirmation, in session
            if close[i] > donchian_upper_aligned[i] and close[i] > ema50_4h_aligned[i] and volume_confirm_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 4h Donchian lower, below 4h EMA50, volume confirmation, in session
            elif close[i] < donchian_lower_aligned[i] and close[i] < ema50_4h_aligned[i] and volume_confirm_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price crosses below 4h EMA50
            if close[i] < ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price crosses above 4h EMA50
            if close[i] > ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals