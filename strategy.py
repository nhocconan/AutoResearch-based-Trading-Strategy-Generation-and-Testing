#!/usr/bin/env python3
# 1h_4H_Donchian_20_1D_EMA34_Trend_Filter
# Hypothesis: Use 4h Donchian channel breakout (20-bar) for trend direction, confirmed by 1d EMA34 trend.
# Enter long when price breaks above 4h Donchian upper band in uptrend (price > 1d EMA34).
# Enter short when price breaks below 4h Donchian lower band in downtrend (price < 1d EMA34).
# Use 1h timeframe for precise entry timing, with 4h/1d for signal direction.
# Add volume confirmation (1.5x 20-bar volume MA) to reduce false breakouts.
# Apply session filter (08-20 UTC) to avoid low-liquidity hours.
# Position size fixed at 0.20 to manage risk and limit trade frequency.
# Target: 15-35 trades/year per symbol to stay within fee-efficient range.

name = "1h_4H_Donchian_20_1D_EMA34_Trend_Filter"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-calculate session filter (08-20 UTC)
    # prices.index is DatetimeIndex, so .hour works directly
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Donchian channel (20-period)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels: upper = 20-period high, lower = 20-period low
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1h timeframe (wait for 4h bar to close)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Get 1d data for trend filter (EMA 34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA 34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA to 1h timeframe (wait for 1d bar to close)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: 20-period volume MA on 1h chart
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Initialize signals array
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Determine warmup period: max of Donchian (20), EMA (34), volume MA (20)
    start_idx = max(20, 34, 20)
    
    # Main loop: process each bar starting from warmup
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any critical values are NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check volume confirmation (1.5x average volume)
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        # Check 1d trend filter
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above 4h Donchian high + uptrend + volume
            if (close[i] > donchian_high_aligned[i]) and uptrend and volume_confirm:
                signals[i] = 0.20
                position = 1
            # Short entry: price breaks below 4h Donchian low + downtrend + volume
            elif (close[i] < donchian_low_aligned[i]) and downtrend and volume_confirm:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price breaks below 4h Donchian low (contrarian exit)
            if close[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price breaks above 4h Donchian high (contrarian exit)
            if close[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals