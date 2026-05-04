#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume spike confirmation
# Uses 1d EMA50 for trend direction to align with higher timeframe momentum
# Uses 12h volume > 2.0x 20-period EMA for strong confirmation to filter weak breakouts
# Designed for 12h timeframe targeting 12-37 trades/year with discrete sizing (0.25)
# Works in bull markets (breakouts with volume in uptrend) and bear markets (breakouts with volume in downtrend)
# Donchian channels provide clear structure with statistical significance for breakouts

name = "12h_Donchian20_Breakout_1dEMA50_Trend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter (higher timeframe than 12h)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 1d data for Donchian channel calculation (20-period)
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels from 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Upper channel: 20-period high
    upper_channel = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Lower channel: 20-period low
    lower_channel = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 12h timeframe (use previous day's levels)
    upper_channel_aligned = align_htf_to_ltf(prices, df_1d, upper_channel)
    lower_channel_aligned = align_htf_to_ltf(prices, df_1d, lower_channel)
    
    # Get 12h data for volume EMA
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h volume EMA(20) for volume confirmation
    vol_12h = df_12h['volume'].values
    vol_ema_20 = pd.Series(vol_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20_aligned = align_htf_to_ltf(prices, df_12h, vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(upper_channel_aligned[i]) or 
            np.isnan(lower_channel_aligned[i]) or np.isnan(vol_ema_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 12h volume > 2.0 x 20-period EMA
        volume_confirmed = volume[i] > (2.0 * vol_ema_20_aligned[i])
        
        if position == 0:
            # Long: price breaks above upper channel + volume confirmation + 1d EMA > EMA50 (uptrend)
            if (close[i] > upper_channel_aligned[i] and volume_confirmed and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower channel + volume confirmation + 1d EMA < EMA50 (downtrend)
            elif (close[i] < lower_channel_aligned[i] and volume_confirmed and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price falls below lower channel (mean reversion) OR 1d EMA < EMA50 (trend change)
            if close[i] < lower_channel_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises above upper channel (mean reversion) OR 1d EMA > EMA50 (trend change)
            if close[i] > upper_channel_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals