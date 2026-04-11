#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + weekly pivot direction + volume confirmation
# - Donchian breakout: price breaks above 20-period high (long) or below 20-period low (short)
# - Weekly pivot direction: price above/below weekly pivot point from prior week
# - Volume confirmation: current volume > 1.5x 20-period average volume
# - Works in both bull (breakouts with volume) and bear (breakdowns with volume) markets
# - Weekly pivot adds structural bias, reducing false breakouts in choppy markets
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits for 6h

name = "6h_1w_donchian_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load weekly data ONCE before loop for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return signals
    
    # Pre-compute weekly pivot point (standard: (H+L+C)/3)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    
    # Pre-compute Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 6h volume SMA (20-period)
    volume_series = pd.Series(volume)
    volume_sma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(pivot_1w_aligned[i]) or np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        
        # Donchian breakout conditions
        breakout_long = price_high > donchian_high[i]  # New 20-period high
        breakout_short = price_low < donchian_low[i]   # New 20-period low
        
        # Weekly pivot direction
        price_above_pivot = price_close > pivot_1w_aligned[i]
        price_below_pivot = price_close < pivot_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Donchian breakout high + price above weekly pivot + volume confirmation
        if breakout_long and price_above_pivot and vol_confirm:
            enter_long = True
        
        # Short: Donchian breakout low + price below weekly pivot + volume confirmation
        if breakout_short and price_below_pivot and vol_confirm:
            enter_short = True
        
        # Exit conditions: opposite Donchian breakout or price crosses weekly pivot
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if Donchian breakdown OR price crosses below weekly pivot
            exit_long = (price_low < donchian_low[i]) or (not price_above_pivot)
        elif position == -1:
            # Exit short if Donchian breakout OR price crosses above weekly pivot
            exit_short = (price_high > donchian_high[i]) or (not price_below_pivot)
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals