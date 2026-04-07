#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Donchian Breakout + 1d Trend + Volume Confirmation
# Hypothesis: Donchian(20) breakouts in direction of 1d EMA(50) trend with volume > 1.5x average capture strong moves.
# Works in bull/bear by trading breakouts with trend filter. Target: 100-200 total trades over 4 years (25-50/year).

name = "4h_donchian_breakout_1d_trend_volume_v2"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Donchian Channel (20) on 4h
    donchian_period = 20
    highest_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_avg[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low or trend changes
            if close[i] < lowest_low[i] or close[i] < ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high or trend changes
            if close[i] > highest_high[i] or close[i] > ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout above Donchian high with uptrend and volume confirmation
            if (close[i] > highest_high[i] and 
                close[i] > ema_50_1d_aligned[i] and
                volume[i] > 1.5 * vol_avg[i]):
                position = 1
                signals[i] = 0.25
            # Breakdown below Donchian low with downtrend and volume confirmation
            elif (close[i] < lowest_low[i] and 
                  close[i] < ema_50_1d_aligned[i] and
                  volume[i] > 1.5 * vol_avg[i]):
                position = -1
                signals[i] = -0.25
    
    return signals