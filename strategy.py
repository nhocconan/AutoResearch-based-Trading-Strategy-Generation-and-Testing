#!/usr/bin/env python3
# 12h_donchian_breakout_1w_trend_volume_v1
# Hypothesis: Donchian breakout with weekly trend and volume confirmation on 12h timeframe.
# Long when price breaks above 20-period Donchian high and price > weekly EMA200 with volume > 1.5x average.
# Short when price breaks below 20-period Donchian low and price < weekly EMA200 with volume > 1.5x average.
# Exit on opposite Donchian break or when volume drops below average.
# Designed to capture strong trends with volume confirmation to reduce whipsaw.
# Target: 50-150 total trades over 4 years (~12-38/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_1w_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly EMA200 for trend filter
    df_1w = get_htf_data(prices, '1w')
    ema_200_1w = pd.Series(df_1w['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(ema_200_1w_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low or volume drops below average
            if close[i] < lowest_low[i] or volume[i] < avg_volume[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high or volume drops below average
            if close[i] > highest_high[i] or volume[i] < avg_volume[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Trend filter: price vs weekly EMA200
            price_above_ema = close[i] > ema_200_1w_aligned[i]
            price_below_ema = close[i] < ema_200_1w_aligned[i]
            
            # Donchian breakout entries
            if close[i] > highest_high[i] and price_above_ema and volume_ok:
                # Additional confirmation: previous close was at or below previous Donchian high
                if i > 0 and close[i-1] <= highest_high[i-1]:
                    position = 1
                    signals[i] = 0.25
            elif close[i] < lowest_low[i] and price_below_ema and volume_ok:
                # Additional confirmation: previous close was at or above previous Donchian low
                if i > 0 and close[i-1] >= lowest_low[i-1]:
                    position = -1
                    signals[i] = -0.25
    
    return signals