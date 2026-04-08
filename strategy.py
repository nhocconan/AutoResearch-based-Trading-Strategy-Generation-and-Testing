#!/usr/bin/env python3
# 4h_donchian_breakout_12h_trend_volume
# Hypothesis: Donchian channel breakout on 4h filtered by 12h EMA trend and volume confirmation.
# Long when price breaks above 20-period Donchian high with uptrend (price > 12h EMA50) and volume > 1.5x average.
# Short when price breaks below 20-period Donchian low with downtrend (price < 12h EMA50) and volume > 1.5x average.
# Designed to capture strong trending moves while avoiding choppy markets. Target: 25-50 trades/year (~100-200 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_12h_trend_volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(avg_volume[i]) or np.isnan(ema_50_12h_aligned[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low OR trend turns against us
            if (close[i] < lowest_low[i]) or (close[i] < ema_50_12h_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR trend turns against us
            if (close[i] > highest_high[i]) or (close[i] > ema_50_12h_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Long entry: price breaks above Donchian high with uptrend and volume confirmation
            if (close[i] > highest_high[i]) and (close[i] > ema_50_12h_aligned[i]) and volume_ok:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Donchian low with downtrend and volume confirmation
            elif (close[i] < lowest_low[i]) and (close[i] < ema_50_12h_aligned[i]) and volume_ok:
                position = -1
                signals[i] = -0.25
    
    return signals