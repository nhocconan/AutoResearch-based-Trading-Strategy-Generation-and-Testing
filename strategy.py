#!/usr/bin/env python3
"""
Hypothesis: 4-hour Donchian breakout with 1-day EMA trend filter and volume confirmation.
Long when price breaks above 4-hour Donchian high (20) + price > 1-day EMA(34) + volume > 1.5x average.
Short when price breaks below 4-hour Donchian low (20) + price < 1-day EMA(34) + volume > 1.5x average.
Exit when price crosses the 1-day EMA(34) in opposite direction.
Designed for moderate trade frequency (~20-40/year) to balance signal quality and fee drag.
Works in bull markets via breakouts and in bear via short breakdowns with trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-day data for EMA trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1-day EMA(34)
    daily_close = df_1d['close'].values
    ema_34 = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate 4-hour Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate average volume for confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(avg_volume[i]) or volume[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Donchian breakout + above daily EMA + volume confirmation
            if (close[i] > donchian_high[i-1] and 
                close[i] > ema_34_aligned[i] and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakdown + below daily EMA + volume confirmation
            elif (close[i] < donchian_low[i-1] and 
                  close[i] < ema_34_aligned[i] and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit when price crosses daily EMA in opposite direction
            exit_signal = False
            if position == 1 and close[i] < ema_34_aligned[i]:
                exit_signal = True
            elif position == -1 and close[i] > ema_34_aligned[i]:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian_Breakout_1dEMA34_Trend_Volume"
timeframe = "4h"
leverage = 1.0