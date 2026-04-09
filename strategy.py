#!/usr/bin/env python3
# 6h_donchian_1w_pullback_volume_v1
# Hypothesis: 6h Donchian(20) breakout with pullback to 20EMA, weekly trend filter from 1w close > 200EMA,
# and volume confirmation (>1.5x 20-period average). Designed for low trade frequency
# (target: 50-150 total trades over 4 years) to avoid fee drag. Works in bull/bear
# by using weekly trend filter (only long in weekly uptrend, short in weekly downtrend)
# and waiting for pullbacks to reduce false breakouts. Uses discrete sizing (±0.25)
# to minimize fee churn.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_1w_pullback_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w HTF data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate weekly trend: close > 200EMA = uptrend, close < 200EMA = downtrend
    weekly_uptrend = close_1w > ema_200_1w_aligned[:len(close_1w)] if len(ema_200_1w_aligned) >= len(close_1w) else close_1w > ema_200_1w_aligned[-1]
    weekly_downtrend = close_1w < ema_200_1w_aligned[:len(close_1w)] if len(ema_200_1w_aligned) >= len(close_1w) else close_1w < ema_200_1w_aligned[-1]
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend.astype(float))
    
    # 6h indicators for entry timing
    # Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 20-period EMA for pullback
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or
            np.isnan(ema_20[i]) or np.isnan(volume_ma[i]) or
            np.isnan(weekly_uptrend_aligned[i]) or np.isnan(weekly_downtrend_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below 20EMA OR weekly trend changes to downtrend
            if close[i] < ema_20[i] or weekly_downtrend_aligned[i] > 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above 20EMA OR weekly trend changes to uptrend
            if close[i] > ema_20[i] or weekly_uptrend_aligned[i] > 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed:
                # Long conditions: weekly uptrend + price breaks above Donchian high + pullback to 20EMA
                if (weekly_uptrend_aligned[i] > 0.5 and 
                    high[i] > high_20[i] and 
                    close[i] <= ema_20[i] * 1.02):  # Allow small overshoot above EMA
                    position = 1
                    signals[i] = 0.25
                # Short conditions: weekly downtrend + price breaks below Donchian low + pullback to 20EMA
                elif (weekly_downtrend_aligned[i] > 0.5 and 
                      low[i] < low_20[i] and 
                      close[i] >= ema_20[i] * 0.98):  # Allow small undershoot below EMA
                    position = -1
                    signals[i] = -0.25
    
    return signals