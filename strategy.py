#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation.
# Uses weekly pivot levels (from 1w data) for long-term trend direction,
# Donchian breakout for entry timing, and volume spike for confirmation.
# Designed to capture strong trend moves while avoiding false breakouts in ranging markets.
# Target: 15-30 trades/year to avoid fee drag.
name = "6h_Donchian20_WeeklyPivot_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points: P = (H+L+C)/3
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    # Weekly trend: price above/below pivot
    trend_1w = pivot_1w  # We'll use this as reference level
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # Get daily data for volume average (20-period SMA)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-period SMA of daily volume
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate Donchian channels (20-period) on 6h data
    # Upper channel: highest high of last 20 periods
    # Lower channel: lowest low of last 20 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need 20 periods for Donchian
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(trend_1w_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        trend_level = trend_1w_aligned[i]
        vol_ma = vol_ma_20_aligned[i]
        upper_channel = donchian_high[i]
        lower_channel = donchian_low[i]
        vol_current = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-day average volume
        vol_confirm = vol_current > 1.5 * vol_ma
        
        if position == 0:
            # Enter long: price breaks above Donchian high AND above weekly pivot AND volume confirmation
            if price > upper_channel and price > trend_level and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low AND below weekly pivot AND volume confirmation
            elif price < lower_channel and price < trend_level and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below Donchian low OR loses volume confirmation
            if price < lower_channel or not vol_confirm:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above Donchian high OR loses volume confirmation
            if price > upper_channel or not vol_confirm:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals