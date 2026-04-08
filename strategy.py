#!/usr/bin/env python3
# 4h_donchian_breakout_12h_trend_volume_v1
# Hypothesis: On 4h timeframe, capture breakouts from Donchian channels (20-period) confirmed by 12h trend (EMA25) and volume spikes (>1.5x average volume).
# Works in bull markets by capturing upside breakouts and in bear markets by capturing downside breakdowns.
# Volume confirmation reduces false breakouts. Trend filter ensures we trade with the higher timeframe momentum.
# Low trade frequency (~20-40/year) minimizes fee drag while maintaining edge.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_12h_trend_volume_v1"
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # 12h EMA25 for trend filter
    ema_25_12h = pd.Series(close_12h).ewm(span=25, adjust=False, min_periods=25).mean().values
    ema_25_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_25_12h)
    
    # Donchian channels (20-period) on 4h
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period) for spike detection
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(vol_avg[i]) or np.isnan(ema_25_12h_aligned[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower band OR trend turns bearish
            if (close[i] < low_min[i]) or (ema_25_12h_aligned[i] < ema_25_12h_aligned[i-1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper band OR trend turns bullish
            if (close[i] > high_max[i]) or (ema_25_12h_aligned[i] > ema_25_12h_aligned[i-1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above Donchian upper band + volume spike + bullish trend
            if (close[i] > high_max[i]) and (volume[i] > 1.5 * vol_avg[i]) and (ema_25_12h_aligned[i] > ema_25_12h_aligned[i-1]):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Donchian lower band + volume spike + bearish trend
            elif (close[i] < low_min[i]) and (volume[i] > 1.5 * vol_avg[i]) and (ema_25_12h_aligned[i] < ema_25_12h_aligned[i-1]):
                position = -1
                signals[i] = -0.25
    
    return signals