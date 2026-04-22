#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian Breakout with 1d EMA Trend Filter and Volume Spike.
Long when price breaks above Donchian Upper Band (20) and price > 1d EMA34 and volume > 1.5x average volume.
Short when price breaks below Donchian Lower Band (20) and price < 1d EMA34 and volume > 1.5x average volume.
Exit when price crosses the Donchian Middle Band (20-period average of high/low).
Uses price channel breakouts for trend following, EMA for trend filter, and volume to confirm institutional participation.
Designed to work in both bull and bear markets by filtering with higher timeframe trend.
"""

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
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    middle_band = (highest_high + lowest_low) / 2.0
    
    # Load 1-day EMA for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike filter: current volume > 1.5x 20-period average
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(ema_34_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above upper band, above 1d EMA, and volume spike
            if (close[i] > highest_high[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower band, below 1d EMA, and volume spike
            elif (close[i] < lowest_low[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price crosses middle band
            exit_signal = False
            
            if position == 1:
                # Exit long: Price falls below middle band
                if close[i] < middle_band[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price rises above middle band
                if close[i] > middle_band[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian_Breakout_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0