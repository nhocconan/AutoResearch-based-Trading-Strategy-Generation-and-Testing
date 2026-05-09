#!/usr/bin/env python3
name = "1D_1W_Donchian20_Volume_Trend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter and Donchian channel
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly Donchian channel (20-period)
    donchian_high = np.full_like(close_1w, np.nan)
    donchian_low = np.full_like(close_1w, np.nan)
    
    for i in range(len(close_1w)):
        if i >= 20:
            donchian_high[i] = np.max(high_1w[i-20:i])
            donchian_low[i] = np.min(low_1w[i-20:i])
    
    # Align weekly Donchian to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Calculate weekly EMA20 for trend filter
    if len(close_1w) >= 20:
        ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    else:
        ema20_1w = np.full_like(close_1w, np.nan)
    
    # Align weekly EMA20 to daily timeframe
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Get daily data for volume confirmation
    # Use 20-period volume EMA for surge detection
    if len(volume) >= 20:
        vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    else:
        vol_ema20 = np.full_like(volume, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data for all indicators
    start_idx = max(20, 20)  # Donchian and volume EMA both need 20 periods
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema20_1w_aligned[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market conditions
        # Uptrend: price above weekly EMA20
        uptrend = close[i] > ema20_1w_aligned[i]
        # Downtrend: price below weekly EMA20
        downtrend = close[i] < ema20_1w_aligned[i]
        # Volume surge: current volume > 1.5x 20-day volume EMA
        volume_surge = volume[i] > vol_ema20[i] * 1.5
        
        if position == 0:
            # Enter long: Uptrend + price breaks above weekly Donchian high + volume surge
            if uptrend and close[i] > donchian_high_aligned[i] and volume_surge:
                signals[i] = 0.25
                position = 1
            # Enter short: Downtrend + price breaks below weekly Donchian low + volume surge
            elif downtrend and close[i] < donchian_low_aligned[i] and volume_surge:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Trend turns down OR price breaks below weekly Donchian low
            if not uptrend or close[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Trend turns up OR price breaks above weekly Donchian high
            if not downtrend or close[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals