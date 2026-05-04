#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction (1w Camarilla) and volume confirmation (>1.5x 20 EMA volume)
# Uses Donchian channels from prior completed 6h bar for structure (breakout = new 20-period high/low)
# Weekly Camarilla pivot direction ensures we only trade in the direction of the higher timeframe trend
# Volume confirmation ensures breakout has sufficient participation
# Discrete sizing 0.25 balances risk and return while minimizing fee churn
# Target: 50-150 total trades over 4 years = 12-37/year for 6h timeframe
# Works in both bull (breakouts continuation) and bear (breakdowns continuation) markets
# BTC/ETH focus: avoids SOL-only bias by requiring HTF trend alignment

name = "6h_Donchian20_1wCamarilla_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla pivot trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:  # Need enough data for pivot calculation
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels (R3, S3, R4, S4)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Pivot point calculation
    pivot = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    
    # Camarilla levels
    r3 = pivot + (range_1w * 1.1 / 4.0)
    s3 = pivot - (range_1w * 1.1 / 4.0)
    r4 = pivot + (range_1w * 1.1 / 2.0)
    s4 = pivot - (range_1w * 1.1 / 2.0)
    
    # Determine weekly trend direction: bullish if close > R3, bearish if close < S3
    weekly_bullish = close_1w > r3
    weekly_bearish = close_1w < s3
    
    # Shift by 1 to use only prior completed weekly bar (no look-ahead)
    weekly_bullish_shifted = np.roll(weekly_bullish, 1)
    weekly_bearish_shifted = np.roll(weekly_bearish, 1)
    weekly_bullish_shifted[0] = False
    weekly_bearish_shifted[0] = False
    
    # Align weekly trend to 6h timeframe
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish_shifted.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish_shifted.astype(float))
    
    # Get 6h data for Donchian channels (prior completed 6h bar)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:  # Need enough data for Donchian calculation
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Calculate Donchian(20) channels: upper = max(high, 20), lower = min(low, 20)
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            result[i] = np.nanmax(arr[i-window+1:i+1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            result[i] = np.nanmin(arr[i-window+1:i+1])
        return result
    
    donchian_upper = rolling_max(high_6h, 20)
    donchian_lower = rolling_min(low_6h, 20)
    
    # Shift by 1 to use only prior completed 6h bar (no look-ahead)
    donchian_upper_shifted = np.roll(donchian_upper, 1)
    donchian_lower_shifted = np.roll(donchian_lower, 1)
    donchian_upper_shifted[0] = np.nan
    donchian_lower_shifted[0] = np.nan
    
    # Align Donchian levels to 6h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_6h, donchian_upper_shifted)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_6h, donchian_lower_shifted)
    
    # Volume confirmation: 20-period EMA of volume on 6h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get weekly trend values (already boolean from alignment)
        weekly_bull = weekly_bullish_aligned[i] > 0.5
        weekly_bear = weekly_bearish_aligned[i] > 0.5
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper + weekly bullish + volume spike
            if close[i] > donchian_upper_aligned[i] and weekly_bull and volume[i] > (1.5 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower + weekly bearish + volume spike
            elif close[i] < donchian_lower_aligned[i] and weekly_bear and volume[i] > (1.5 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian lower OR weekly turns bearish
            if close[i] < donchian_lower_aligned[i] or weekly_bear:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian upper OR weekly turns bullish
            if close[i] > donchian_upper_aligned[i] or weekly_bull:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals