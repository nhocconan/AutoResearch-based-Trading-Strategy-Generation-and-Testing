#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Uses Donchian channels from daily timeframe for structural breakout levels
# 1w EMA50 provides weekly trend filter to avoid counter-trend trades
# Volume spike (>1.8x 20-period EMA volume) confirms institutional participation
# Discrete sizing 0.25 targets 30-80 total trades over 4 years (7-20/year) for 1d timeframe
# Works in bull markets (breakouts with uptrend) and bear markets (breakdowns with downtrend)
# 1d timeframe minimizes fee drag while capturing multi-week moves

name = "1d_Donchian20_1wEMA50_VolumeConfirm"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need enough data for Donchian calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels from prior completed 1d bar
    # Upper = max(high of last 20 days), Lower = min(low of last 20 days)
    donchian_upper_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_lower_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Shift to use prior completed 1d bar (avoid look-ahead)
    donchian_upper_1d_shifted = np.roll(donchian_upper_1d, 1)
    donchian_lower_1d_shifted = np.roll(donchian_lower_1d, 1)
    donchian_upper_1d_shifted[0] = np.nan
    donchian_lower_1d_shifted[0] = np.nan
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need enough data for EMA50 calculation
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA(50) trend filter from prior completed 1w bar
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_shifted = np.roll(ema_50_1w, 1)
    ema_50_1w_shifted[0] = np.nan
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align HTF indicators to 1d timeframe (wait for completed bars)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper_1d_shifted)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower_1d_shifted)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w_shifted)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian Upper AND price > 1w EMA50 AND volume spike
            if close[i] > donchian_upper_aligned[i] and close[i] > ema_50_1w_aligned[i] and volume[i] > (1.8 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian Lower AND price < 1w EMA50 AND volume spike
            elif close[i] < donchian_lower_aligned[i] and close[i] < ema_50_1w_aligned[i] and volume[i] > (1.8 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian Lower OR price crosses below 1w EMA50
            if close[i] < donchian_lower_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian Upper OR price crosses above 1w EMA50
            if close[i] > donchian_upper_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals