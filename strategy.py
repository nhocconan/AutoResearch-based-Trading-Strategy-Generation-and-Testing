#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout with volume confirmation and 1d EMA50 trend filter
# - Long when price breaks above 4h Donchian upper band with volume spike and price above 1d EMA50
# - Short when price breaks below 4h Donchian lower band with volume spike and price below 1d EMA50
# - Exit when price crosses below/above 1d EMA50
# - Designed to capture trend continuation in strong markets while filtering false breakouts
# - Target: 60-150 total trades over 4 years (15-37/year) with 0.20 position sizing

name = "1h_Donchian_Breakout_4hUpperLower_1dEMA50_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    donchian_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align 4h Donchian bands to 1h timeframe
    donchian_upper_1h = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_1h = align_htf_to_ltf(prices, df_4h, donchian_lower)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_1h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter (1h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma_20)  # Volume confirmation
    
    # Session filter: 08:00-20:00 UTC (reduce noise outside active hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if any critical value is NaN
        if (np.isnan(donchian_upper_1h[i]) or np.isnan(donchian_lower_1h[i]) or 
            np.isnan(ema_50_1d_1h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Apply session filter
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 4h Donchian upper with volume spike and above 1d EMA50
            if high[i] > donchian_upper_1h[i] and volume_spike[i] and close[i] > ema_50_1d_1h[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h Donchian lower with volume spike and below 1d EMA50
            elif low[i] < donchian_lower_1h[i] and volume_spike[i] and close[i] < ema_50_1d_1h[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price crosses below 1d EMA50
            if close[i] < ema_50_1d_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price crosses above 1d EMA50
            if close[i] > ema_50_1d_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals