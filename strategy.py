#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 1d EMA200 trend filter.
# Uses 1d EMA200 for long-term trend to capture major trend direction, reducing false breakouts.
# Volume > 1.8x EMA20 volume filter ensures institutional participation. Designed for both bull and bear.
name = "4h_Donchian20_EMA200_Volume_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 250:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for EMA200 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Daily data for Donchian channels (20-day period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Donchian channels: 20-period high and low
    # Using pandas rolling for clarity and proper min_periods
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Use previous day's levels (shift by 1 to avoid look-ahead)
    donchian_high_shifted = np.roll(donchian_high, 1)
    donchian_low_shifted = np.roll(donchian_low, 1)
    donchian_high_shifted[0] = np.nan
    donchian_low_shifted[0] = np.nan
    
    # Align to 4h timeframe
    donchian_high_4h = align_htf_to_ltf(prices, df_1d, donchian_high_shifted)
    donchian_low_4h = align_htf_to_ltf(prices, df_1d, donchian_low_shifted)
    
    # 1d EMA200 trend filter
    ema_200_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Volume spike filter: volume > 1.8x EMA20 volume
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (1.8 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(donchian_high_4h[i]) or np.isnan(donchian_low_4h[i]) or
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above 20-day Donchian high with volume spike and above 1d EMA200
            if (price > donchian_high_4h[i] and vol_spike[i] and price > ema_200_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-day Donchian low with volume spike and below 1d EMA200
            elif (price < donchian_low_4h[i] and vol_spike[i] and price < ema_200_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below 20-day Donchian low (mean reversion)
            if price < donchian_low_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above 20-day Donchian high (mean reversion)
            if price > donchian_high_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals