#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with weekly pivot direction and volume confirmation
# Uses weekly high/low as dynamic support/resistance. Breakouts above weekly high or below weekly low
# are traded only when confirmed by volume and aligned with weekly trend (price > weekly EMA20 for longs,
# price < weekly EMA20 for shorts). Works in bull markets (breakouts up with trend) and bear markets
# (breakouts down with trend). Target: 50-150 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data for Donchian channels and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 20-period Donchian channels on weekly
    # Donchian High = max(high_1w over last 20 weeks)
    # Donchian Low = min(low_1w over last 20 weeks)
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Calculate weekly EMA20 for trend filter
    ema20_1w = pd.Series(close_1w).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Align weekly indicators to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size (25% of capital)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema20_1w_aligned[i])):
            continue
        
        # Long entry: price breaks above weekly Donchian high + volume confirmation + bullish trend
        if (close[i] > donchian_high_aligned[i] and
            volume[i] > 1.5 * np.median(volume[max(0, i-10):i+1]) and
            close[i] > ema20_1w_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below weekly Donchian low + volume confirmation + bearish trend
        elif (close[i] < donchian_low_aligned[i] and
              volume[i] > 1.5 * np.median(volume[max(0, i-10):i+1]) and
              close[i] < ema20_1w_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse breakout or trend reversal
        elif position == 1 and (close[i] < donchian_low_aligned[i] or close[i] < ema20_1w_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > donchian_high_aligned[i] or close[i] > ema20_1w_aligned[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_WeeklyDonchian_Breakout_Trend_Volume"
timeframe = "6h"
leverage = 1.0