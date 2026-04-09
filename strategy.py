#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d volume confirmation and weekly trend filter
# - Uses 6h Donchian channel (20-period) for breakout signals in direction of 1w trend
# - Confirms with 1d volume > 1.8x its 20-period average (strong participation)
# - 1w trend filter: price above/below 50-period EMA on weekly timeframe
# - Position size: 0.25 (25% of capital) to balance return and drawdown
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Donchian breakouts capture strong momentum moves; volume confirmation reduces false signals
# - Weekly trend filter ensures we only trade in direction of higher timeframe momentum
# - Works in both bull and bear markets by following weekly trend

name = "6h_1d_1w_donchian_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 60 or len(df_1w) < 60:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    volume_1d = df_1d['volume'].values
    close_1d = df_1d['close'].values
    
    # 1d Volume > 1.8x 20-period average (stricter for fewer trades)
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (1.8 * avg_volume_20)
    
    # Pre-compute 1w indicators
    close_1w = df_1w['close'].values
    # 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d and 1w indicators to 6h
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 6h price data for Donchian calculation
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Pre-compute 6h Donchian channels (20-period)
    # Donchian Upper = max(high, lookback=20)
    # Donchian Lower = min(low, lookback=20)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(volume_spike_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            if high[i] > highest_since_entry:
                highest_since_entry = high[i]
            
            # Exit conditions: price retraces to Donchian Lower (20-period)
            if low[i] <= lowest_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            if low[i] < lowest_since_entry:
                lowest_since_entry = low[i]
            
            # Exit conditions: price retraces to Donchian Upper (20-period)
            if high[i] >= highest_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with volume confirmation and weekly trend filter
            # Long: price breaks above Donchian Upper + volume spike + price above weekly EMA
            # Short: price breaks below Donchian Lower + volume spike + price below weekly EMA
            if (high[i] >= highest_20[i] and    # Break above Donchian Upper
                volume_spike_1d_aligned[i] and   # Volume confirmation
                close[i] > ema_50_1w_aligned[i]): # Weekly trend filter (bullish)
                position = 1
                entry_price = high[i]
                highest_since_entry = high[i]
                lowest_since_entry = high[i]  # Initialize for shorts
                signals[i] = 0.25
            elif (low[i] <= lowest_20[i] and    # Break below Donchian Lower
                  volume_spike_1d_aligned[i] and # Volume confirmation
                  close[i] < ema_50_1w_aligned[i]): # Weekly trend filter (bearish)
                position = -1
                entry_price = low[i]
                highest_since_entry = low[i]  # Initialize for longs
                lowest_since_entry = low[i]
                signals[i] = -0.25
    
    return signals