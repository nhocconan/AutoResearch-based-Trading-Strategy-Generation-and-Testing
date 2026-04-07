#!/usr/bin/env python3
"""
1d_donchian_breakout_1w_trend_volume_v1
Hypothesis: On daily timeframe, enter long when price breaks above 20-day Donchian high with weekly trend confirmation and above-average volume, enter short when price breaks below 20-day Donchian low with weekly trend confirmation and above-average volume. Exit when price crosses the 20-day Donchian mean. Uses weekly EMA trend filter to avoid counter-trend trades. Designed for 20-40 trades/year to minimize fee decay while capturing breakouts in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_breakout_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 20-day Donchian channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mean = (donchian_high + donchian_low) / 2.0
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate weekly EMA for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mean[i]) or np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: above average volume
        vol_ok = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below Donchian mean
            if close[i] < donchian_mean[i] and close[i-1] >= donchian_mean[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian mean
            if close[i] > donchian_mean[i] and close[i-1] <= donchian_mean[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Long: price breaks above Donchian high with weekly uptrend
                if close[i] > donchian_high[i] and close[i-1] <= donchian_high[i-1] and ema_20_1w_aligned[i] > close_1w[-1] if len(close_1w) > 0 else False:
                    # Actually need to compare with previous weekly close for trend
                    # Simplified: weekly EMA > previous weekly close (approximated)
                    if i >= 20:  # Ensure we have weekly data
                        # Get the weekly EMA value at this point (already aligned)
                        # Compare with weekly close from same week (approximated by using the aligned value vs price)
                        # Better approach: check if weekly EMA is sloping up
                        pass  # Simplify for now
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below Donchian low with weekly downtrend
                elif close[i] < donchian_low[i] and close[i-1] >= donchian_low[i-1] and ema_20_1w_aligned[i] < close_1w[-1] if len(close_1w) > 0 else False:
                    if i >= 20:
                        pass
                    position = -1
                    signals[i] = -0.25
    
    # Fix the trend comparison - use proper weekly trend
    # Re-implement with correct trend logic
    
    # Recalculate for clarity
    signals = np.zeros(n)
    position = 0
    
    # Pre-calculate weekly trend: 1 if EMA rising, -1 if falling
    weekly_trend = np.zeros_like(ema_20_1w_aligned)
    for i in range(1, len(weekly_trend)):
        if not np.isnan(ema_20_1w_aligned[i]) and not np.isnan(ema_20_1w_aligned[i-1]):
            if ema_20_1w_aligned[i] > ema_20_1w_aligned[i-1]:
                weekly_trend[i] = 1
            else:
                weekly_trend[i] = -1
        else:
            weekly_trend[i] = weekly_trend[i-1] if i > 0 else 0
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mean[i]) or np.isnan(weekly_trend[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: above average volume
        vol_ok = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below Donchian mean
            if close[i] < donchian_mean[i] and close[i-1] >= donchian_mean[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian mean
            if close[i] > donchian_mean[i] and close[i-1] <= donchian_mean[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Long: price breaks above Donchian high with weekly uptrend
                if close[i] > donchian_high[i] and close[i-1] <= donchian_high[i-1] and weekly_trend[i] == 1:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below Donchian low with weekly downtrend
                elif close[i] < donchian_low[i] and close[i-1] >= donchian_low[i-1] and weekly_trend[i] == -1:
                    position = -1
                    signals[i] = -0.25
    
    return signals