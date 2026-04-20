#!/usr/bin/env python3
# 4h_Donchian_Breakout_VolumeTrend_Filter
# Hypothesis: Donchian channel breakouts with volume confirmation and trend filter capture strong momentum moves.
# Works in bull markets by catching breakouts to new highs and in bear markets by catching breakdowns to new lows.
# Volume confirmation ensures institutional participation, while 4h EMA20 filter avoids counter-trend trades.
# Designed for 4h timeframe to balance trade frequency and signal quality, targeting 20-40 trades/year.

name = "4h_Donchian_Breakout_VolumeTrend_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d EMA20 for trend filter
    ema_20_1d = pd.Series(df_1d['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Donchian channel (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for Donchian channel to be ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(ema_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high + volume + uptrend (price > 1d EMA20)
            if close[i] > high_20[i] and volume_filter[i] and close[i] > ema_20_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + volume + downtrend (price < 1d EMA20)
            elif close[i] < low_20[i] and volume_filter[i] and close[i] < ema_20_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below Donchian low (reversal)
            if close[i] < low_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above Donchian high (reversal)
            if close[i] > high_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals