#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Donchian(20) breakout with 1-week trend filter and volume confirmation
# In bull market (weekly close > weekly EMA10): buy breakouts above Donchian(20) high
# In bear market (weekly close < weekly EMA10): sell breakdowns below Donchian(20) low
# Volume confirmation: require volume > 2x 20-day average to filter false breakouts
# Exit: opposite Donchian band touch or volatility contraction
# Designed to capture major trends while avoiding whipsaws in ranging markets
# Target: 30-100 total trades over 4 years (7-25/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 10-period EMA on weekly timeframe for trend filter
    ema10_1w = pd.Series(close_1w).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema10_1w)
    
    # Load daily data for Donchian channels and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Align daily data to lower timeframe (assuming 1d is HTF for intraday, but here we use 1d as primary)
    # Since we're using 1d as primary timeframe, we need to align to itself (no change)
    # But we'll still use the arrays directly for consistency
    high = high_1d
    low = low_1d
    close = close_1d
    volume = volume_1d
    
    # Calculate 20-period Donchian channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume filter: volume > 2x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if NaN in indicators
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(ema10_1w_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market trend from weekly EMA
        # Note: We use weekly close for trend, but need to get the aligned weekly close
        # Since we don't have weekly close aligned, we approximate using the EMA alignment
        # A better approach would be to align weekly close, but EMA trend is sufficient
        is_bull = close[i] > ema10_1w_aligned[i]  # Simplified: price above weekly EMA10
        is_bear = close[i] < ema10_1w_aligned[i]
        
        # Volume confirmation
        has_volume = vol_filter[i]
        
        if position == 0:
            # Enter long: bullish trend + breakout above Donchian high + volume
            if is_bull and has_volume and close[i] > highest_high[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish trend + breakdown below Donchian low + volume
            elif is_bear and has_volume and close[i] < lowest_low[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: touch Donchian low or trend change to bear
            if close[i] < lowest_low[i] or is_bear:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: touch Donchian high or trend change to bull
            if close[i] > highest_high[i] or is_bull:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_WeeklyTrendFilter_Volume"
timeframe = "1d"
leverage = 1.0