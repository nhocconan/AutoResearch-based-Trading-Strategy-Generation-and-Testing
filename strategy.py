#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Donchian Breakout with Volume Confirmation
# Hypothesis: Donchian(20) breakouts on 12h with volume confirmation capture breakout moves.
# Uses 1d trend filter to avoid counter-trend trades. Works in bull/bear by following trend.
# Target: 25-40 trades/year (100-160 total over 4 years) to minimize fee drag.

name = "12h_donchian_breakout_1d_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian(20) on 12h high/low
    highest_high = pd.Series(close).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(close).rolling(window=20, min_periods=20).min().values
    
    # Volume average for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(vol_avg[i]) or np.isnan(ema_50_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian lower band or trend changes
            if close[i] < lowest_low[i] or close[i] < ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian upper band or trend changes
            if close[i] > highest_high[i] or close[i] > ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average
            volume_confirm = volume[i] > 1.5 * vol_avg[i]
            
            # Long: price breaks above Donchian upper band in uptrend
            if close[i] > highest_high[i] and close[i] > ema_50_1d_aligned[i] and volume_confirm:
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian lower band in downtrend
            elif close[i] < lowest_low[i] and close[i] < ema_50_1d_aligned[i] and volume_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals