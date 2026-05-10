#!/usr/bin/env python3
# 4H_12H_Donchian20_Breakout_12hTrend_Volume
# Hypothesis: On 4h timeframe, enter long when price breaks above Donchian(20) high with 12h uptrend (EMA50) and volume confirmation; enter short when price breaks below Donchian(20) low with 12h downtrend and volume confirmation.
# Uses 12h EMA for trend filter to avoid counter-trend trades, volume spike to confirm breakout strength.
# Designed for low trade frequency (~20-40/year) to minimize fee drag and work in both bull/bear markets via trend following.

name = "4H_12H_Donchian20_Breakout_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter (EMA50) and Donchian calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 12h EMA50 for trend filter
    if len(close_12h) >= 50:
        close_12h_series = pd.Series(close_12h)
        ema_50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    else:
        ema_50_12h = np.full_like(close_12h, np.nan)
    
    # Calculate 12h Donchian channels (20-period)
    if len(high_12h) >= 20 and len(low_12h) >= 20:
        high_12h_series = pd.Series(high_12h)
        low_12h_series = pd.Series(low_12h)
        donchian_high_12h = high_12h_series.rolling(window=20, min_periods=20).max().values
        donchian_low_12h = low_12h_series.rolling(window=20, min_periods=20).min().values
    else:
        donchian_high_12h = np.full_like(high_12h, np.nan)
        donchian_low_12h = np.full_like(low_12h, np.nan)
    
    # Align 12h indicators to 4h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    donchian_high_12h_aligned = align_htf_to_ltf(prices, df_12h, donchian_high_12h)
    donchian_low_12h_aligned = align_htf_to_ltf(prices, df_12h, donchian_low_12h)
    
    # Calculate 4h Donchian channels (20-period) for entry signals
    if len(high) >= 20 and len(low) >= 20:
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        donchian_high_4h = high_series.rolling(window=20, min_periods=20).max().values
        donchian_low_4h = low_series.rolling(window=20, min_periods=20).min().values
    else:
        donchian_high_4h = np.full_like(high, np.nan)
        donchian_low_4h = np.full_like(low, np.nan)
    
    # Calculate 4h volume moving average for volume confirmation
    if len(volume) >= 20:
        volume_series = pd.Series(volume)
        volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    else:
        volume_ma_20 = np.full_like(volume, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data for all indicators
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(donchian_high_12h_aligned[i]) or 
            np.isnan(donchian_low_12h_aligned[i]) or np.isnan(donchian_high_4h[i]) or 
            np.isnan(donchian_low_4h[i]) or np.isnan(volume_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 12h trend direction
        uptrend_12h = close_12h[i] > ema_50_12h[i] if not np.isnan(close_12h[i]) and not np.isnan(ema_50_12h[i]) else False
        downtrend_12h = close_12h[i] < ema_50_12h[i] if not np.isnan(close_12h[i]) and not np.isnan(ema_50_12h[i]) else False
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * volume_ma_20[i]
        
        if position == 0:
            # Enter long: 12h uptrend + price breaks above 4h Donchian high + volume confirmation
            if uptrend_12h and close[i] > donchian_high_4h[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Enter short: 12h downtrend + price breaks below 4h Donchian low + volume confirmation
            elif downtrend_12h and close[i] < donchian_low_4h[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: 12h trend turns down OR price breaks below 4h Donchian low
            if not uptrend_12h or close[i] < donchian_low_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: 12h trend turns up OR price breaks above 4h Donchian high
            if not downtrend_12h or close[i] > donchian_high_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals