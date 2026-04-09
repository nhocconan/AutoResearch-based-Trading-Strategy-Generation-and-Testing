#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA trend filter and volume confirmation
# Donchian breakout captures strong momentum moves in both bull and bear markets
# 1d EMA(50) determines primary trend direction - only take breakouts in trend direction
# Volume confirmation ensures breakout authenticity (avoids false breakouts)
# Uses discrete position sizing (0.25) to minimize fee churn
# Target: 75-200 total trades over 4 years (19-50/year)

name = "4h_1d_donchian_ema_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d average volume (20-period) for confirmation
    volume_1d = df_1d['volume'].values
    avg_volume_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    
    # Calculate 4h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(avg_volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.3x 1d average volume
        volume_confirmed = volume[i] > 1.3 * avg_volume_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower band
            if close[i] < lowest_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper band
            if close[i] > highest_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: Donchian breakout in direction of 1d EMA trend with volume confirmation
            if close[i] > highest_high[i] and ema_50_1d_aligned[i] < close_1d[-1] if len(close_1d) > 0 else True and volume_confirmed:
                # Only go long if 1d EMA is trending up (current price above EMA)
                if close_1d[-1] > ema_50_1d_aligned[i] if len(close_1d) > 0 else True:
                    position = 1
                    signals[i] = 0.25
            elif close[i] < lowest_low[i] and ema_50_1d_aligned[i] > close_1d[-1] if len(close_1d) > 0 else True and volume_confirmed:
                # Only go short if 1d EMA is trending down (current price below EMA)
                if close_1d[-1] < ema_50_1d_aligned[i] if len(close_1d) > 0 else True:
                    position = -1
                    signals[i] = -0.25
    
    return signals