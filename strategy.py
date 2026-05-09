#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with volume confirmation and 1d EMA50 trend filter.
# Uses price channel breakouts for trend following, volume confirms institutional interest,
# 1d EMA50 filters for trend direction to avoid counter-trend entries.
# Designed for 12h timeframe to target 50-150 total trades over 4 years (12-37/year).
# Works in both bull and bear markets by following higher timeframe trend.
name = "12h_Donchian20_Volume_1dEMA50_Trend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for EMA50 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily data for Donchian channels (using previous day's data to avoid look-ahead)
    df_1d_donch = get_htf_data(prices, '1d')
    if len(df_1d_donch) < 20:
        return np.zeros(n)
    
    high_1d = df_1d_donch['high'].values
    low_1d = df_1d_donch['low'].values
    
    # Calculate Donchian channels: 20-period high and low
    # Using pandas rolling for efficiency
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    donch_high = high_series.rolling(window=20, min_periods=20).max().values
    donch_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Use previous day's levels (shift by 1 to avoid look-ahead)
    donch_high_shifted = np.roll(donch_high, 1)
    donch_low_shifted = np.roll(donch_low, 1)
    donch_high_shifted[0] = np.nan
    donch_low_shifted[0] = np.nan
    
    # Align to 12h timeframe
    donch_high_12h = align_htf_to_ltf(prices, df_1d_donch, donch_high_shifted)
    donch_low_12h = align_htf_to_ltf(prices, df_1d_donch, donch_low_shifted)
    
    # 1d EMA50 trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike filter: volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(donch_high_12h[i]) or np.isnan(donch_low_12h[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above Donchian high with volume spike and above 1d EMA50
            if (price > donch_high_12h[i] and vol_spike[i] and price > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume spike and below 1d EMA50
            elif (price < donch_low_12h[i] and vol_spike[i] and price < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below Donchian low (mean reversion to support)
            if price < donch_low_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above Donchian high (mean reversion to resistance)
            if price > donch_high_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals