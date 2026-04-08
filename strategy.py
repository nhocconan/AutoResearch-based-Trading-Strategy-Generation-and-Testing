#!/usr/bin/env python3
# 4h_trend_volume_breakout_12h
# Hypothesis: On 4h timeframe, enter long when price breaks above 20-period Donchian high with 12h EMA50 uptrend and volume > 1.5x average.
# Enter short when price breaks below 20-period Donchian low with 12h EMA50 downtrend and volume > 1.5x average.
# Exit when price crosses the 12-period EMA (4h) or Donchian middle band.
# Uses volume confirmation and trend filter to reduce false breakouts.
# Target: 20-50 trades per year per symbol (~80-200 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_trend_volume_breakout_12h"
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
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate 4h EMA12 for exit
    ema_12 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(ema_12[i]) or 
            np.isnan(avg_volume[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below EMA12 OR Donchian middle band
            if (close[i] < ema_12[i]) or (close[i] < donchian_mid[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above EMA12 OR Donchian middle band
            if (close[i] > ema_12[i]) or (close[i] > donchian_mid[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Breakout entries
            if (close[i] > donchian_high[i]) and (close[i] > ema_50_12h_aligned[i]) and volume_ok:
                position = 1
                signals[i] = 0.25
            elif (close[i] < donchian_low[i]) and (close[i] < ema_50_12h_aligned[i]) and volume_ok:
                position = -1
                signals[i] = -0.25
    
    return signals