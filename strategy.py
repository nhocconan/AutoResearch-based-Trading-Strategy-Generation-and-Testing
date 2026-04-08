#!/usr/bin/env python3
# 1h_volume_breakout_4h1d_trend_v1
# Hypothesis: Breakout of 1h price range with volume confirmation, filtered by 4h and 1d trend.
# Long when price breaks above 1h high with volume > 2x average and 4h/1d uptrend.
# Short when price breaks below 1h low with volume > 2x average and 4h/1d downtrend.
# Exit when price returns to 1h midpoint or opposite signal.
# Designed to work in both bull and bear markets by capturing breakouts with trend confirmation.
# Target: 20-40 trades/year to minimize fee decay while capturing strong directional moves.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_volume_breakout_4h1d_trend_v1"
timeframe = "1h"
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
    
    # Get 4h data for trend filter (calculate once before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 1:
        return np.zeros(n)
    
    # 4h trend: close > open = uptrend, close < open = downtrend
    open_4h = df_4h['open'].values
    close_4h = df_4h['close'].values
    trend_4h_up = close_4h > open_4h
    trend_4h_down = close_4h < open_4h
    
    # Align 4h trend to 1h chart
    trend_4h_up_aligned = align_htf_to_ltf(prices, df_4h, trend_4h_up.astype(float))
    trend_4h_down_aligned = align_htf_to_ltf(prices, df_4h, trend_4h_down.astype(float))
    
    # Get 1d data for trend filter (calculate once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # 1d trend: close > open = uptrend, close < open = downtrend
    open_1d = df_1d['open'].values
    close_1d = df_1d['close'].values
    trend_1d_up = close_1d > open_1d
    trend_1d_down = close_1d < open_1d
    
    # Align 1d trend to 1h chart
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down.astype(float))
    
    # 1h range: calculate rolling high/low
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    midpoint = (highest_high + lowest_low) / 2
    
    # Volume: 20-period average
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = lookback
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(midpoint[i]) or \
           np.isnan(trend_4h_up_aligned[i]) or np.isnan(trend_4h_down_aligned[i]) or \
           np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i]) or \
           np.isnan(avg_volume[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to 1h midpoint or opposite signal
            if close[i] <= midpoint[i] or \
               (close[i] >= lowest_low[i] and volume[i] > 2.0 * avg_volume[i] and trend_4h_down_aligned[i] and trend_1d_down_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price returns to 1h midpoint or opposite signal
            if close[i] >= midpoint[i] or \
               (close[i] <= highest_high[i] and volume[i] > 2.0 * avg_volume[i] and trend_4h_up_aligned[i] and trend_1d_up_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 2x average volume
            volume_ok = volume[i] > 2.0 * avg_volume[i]
            
            # Long entry: price breaks above 1h high with volume and 4h/1d uptrend
            if close[i] > highest_high[i] and volume_ok and trend_4h_up_aligned[i] and trend_1d_up_aligned[i]:
                position = 1
                signals[i] = 0.20
            # Short entry: price breaks below 1h low with volume and 4h/1d downtrend
            elif close[i] < lowest_low[i] and volume_ok and trend_4h_down_aligned[i] and trend_1d_down_aligned[i]:
                position = -1
                signals[i] = -0.20
    
    return signals