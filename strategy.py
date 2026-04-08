#!/usr/bin/env python3
# 12h_donchian_breakout_1d_trend_volume_v1
# Hypothesis: Donchian channel breakout on 12h with 1d trend filter and volume confirmation.
# Long when price breaks above 20-period Donchian high, 1d EMA50 trending up, and volume > 1.5x average.
# Short when price breaks below 20-period Donchian low, 1d EMA50 trending down, and volume > 1.5x average.
# Exit on opposite breakout or when volume drops below average.
# Target: 50-150 total trades over 4 years (~12-37/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

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
    
    # Calculate Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(avg_volume[i]) or np.isnan(ema_50_1d_aligned[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low or volume drops below average
            if close[i] < donchian_low[i] or volume[i] < avg_volume[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high or volume drops below average
            if close[i] > donchian_high[i] or volume[i] < avg_volume[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Donchian breakout entries with 1d trend filter
            if close[i] > donchian_high[i] and volume_ok:
                # Additional confirmation: 1d EMA50 trending up (current > previous)
                if i > 0 and ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]:
                    position = 1
                    signals[i] = 0.25
            elif close[i] < donchian_low[i] and volume_ok:
                # Additional confirmation: 1d EMA50 trending down (current < previous)
                if i > 0 and ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]:
                    position = -1
                    signals[i] = -0.25
    
    return signals