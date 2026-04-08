#!/usr/bin/env python3
# 4h_donchian_breakout_1d_trend_volume_v6
# Hypothesis: 4h Donchian(20) breakout with 1d EMA(50) trend filter and volume confirmation.
# Long when price breaks above upper Donchian channel with price > 1d EMA and volume > 1.5x average.
# Short when price breaks below lower Donchian channel with price < 1d EMA and volume > 1.5x average.
# Exit on opposite Donchian breakout or when volume drops below average.
# Uses 1d EMA for trend filter to capture multi-day trends while minimizing whipsaw in ranging markets.
# Target: 150-250 total trades over 4 years (~38-63/year) with strong trend capture.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_1d_trend_volume_v6"
timeframe = "4h"
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
    
    # Calculate Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(avg_volume[i]) or np.isnan(ema_50_1d_aligned[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below lower Donchian or volume drops below average
            if close[i] < low_20[i] or volume[i] < avg_volume[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price breaks above upper Donchian or volume drops below average
            if close[i] > high_20[i] or volume[i] < avg_volume[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Donchian breakout entries with 1d EMA trend filter
            if (close[i] > high_20[i]) and volume_ok and (close[i] > ema_50_1d_aligned[i]):
                # Additional confirmation: previous close was at or below upper channel
                if i > 0 and close[i-1] <= high_20[i-1]:
                    position = 1
                    signals[i] = 0.30
            elif (close[i] < low_20[i]) and volume_ok and (close[i] < ema_50_1d_aligned[i]):
                # Additional confirmation: previous close was at or above lower channel
                if i > 0 and close[i-1] >= low_20[i-1]:
                    position = -1
                    signals[i] = -0.30
    
    return signals