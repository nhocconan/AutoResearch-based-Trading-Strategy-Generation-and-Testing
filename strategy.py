#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian channel breakout with 1d trend filter and volume confirmation
# Uses Donchian breakout for entry, 1d EMA for trend filter, and volume spike for confirmation
# Designed to work in both bull and bear markets by requiring strong trend alignment and volume confirmation
# Target: 12-37 trades/year, focused on high-probability breakouts with confirmation
name = "12h_donchian_breakout_1d_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian calculation and trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d EMA for trend filter (50-period)
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d volume SMA for volume context (20-period)
    vol_sma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d Donchian channels (20-period)
    high_max_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_min_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_1d[i]) or np.isnan(volume_1d[i]) or 
            np.isnan(vol_sma_1d[i]) or np.isnan(high_max_1d[i]) or 
            np.isnan(low_min_1d[i])):
            signals[i] = 0.0
            continue
        
        # Get aligned 1d values for current 12h bar
        ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)[i]
        vol_sma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_1d)[i]
        high_max_1d_aligned = align_htf_to_ltf(prices, df_1d, high_max_1d)[i]
        low_min_1d_aligned = align_htf_to_ltf(prices, df_1d, low_min_1d)[i]
        
        # Trend filter: price above/below 50 EMA on 1d
        uptrend = close[i] > ema_1d_aligned
        downtrend = close[i] < ema_1d_aligned
        
        # Volume filter: current volume above 2.5x 1d average volume (more selective)
        volume_filter = volume[i] > (vol_sma_1d_aligned * 2.5)
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low OR trend reversal
            if close[i] < low_min_1d_aligned or not uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR trend reversal
            if close[i] > high_max_1d_aligned or not downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat, look for entry
            # Long: price breaks above Donchian high + uptrend + volume filter
            if close[i] > high_max_1d_aligned and uptrend and volume_filter:
                position = 1
                signals[i] = 0.30
            # Short: price breaks below Donchian low + downtrend + volume filter
            elif close[i] < low_min_1d_aligned and downtrend and volume_filter:
                position = -1
                signals[i] = -0.30
    
    return signals