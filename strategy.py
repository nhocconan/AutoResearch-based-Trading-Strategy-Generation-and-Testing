#!/usr/bin/env python3
"""
Hypothesis: 4-hour Donchian breakout with 12-hour EMA trend filter and volume confirmation.
Long when price breaks above Donchian upper channel and 12h EMA(50) > EMA(100) and volume > 1.5x average.
Short when price breaks below Donchian lower channel and 12h EMA(50) < EMA(100) and volume > 1.5x average.
Exit when price crosses Donchian midline or volume drops below average.
Donchian channels provide trend-following structure; volume filter ensures institutional participation;
12h EMA trend filter avoids counter-trend trades. Works in bull markets via breakouts and in bear
markets via short breakdowns. Designed for low trade frequency to minimize fee drag.
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
    donchian_len = 20
    upper = pd.Series(high).rolling(window=donchian_len, min_periods=donchian_len).max().values
    lower = pd.Series(low).rolling(window=donchian_len, min_periods=donchian_len).min().values
    middle = (upper + lower) / 2.0
    
    # Load 12h data for EMA trend and volume filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # 12h EMA(50) and EMA(100) for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_100_12h = pd.Series(close_12h).ewm(span=100, adjust=False, min_periods=100).mean().values
    
    # 12h average volume for filter
    avg_vol_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Align 12h indicators to 4h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    ema_100_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_100_12h)
    avg_vol_12h_aligned = align_htf_to_ltf(prices, df_12h, avg_vol_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(donchian_len, n):
        # Skip if data not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(ema_100_12h_aligned[i]) or
            np.isnan(avg_vol_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above upper Donchian, 12h EMA50 > EMA100, volume > 1.5x avg
            if (close[i] > upper[i] and 
                ema_50_12h_aligned[i] > ema_100_12h_aligned[i] and
                volume[i] > 1.5 * avg_vol_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian, 12h EMA50 < EMA100, volume > 1.5x avg
            elif (close[i] < lower[i] and 
                  ema_50_12h_aligned[i] < ema_100_12h_aligned[i] and
                  volume[i] > 1.5 * avg_vol_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price falls below Donchian midline
                if close[i] < middle[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price rises above Donchian midline
                if close[i] > middle[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian_Breakout_12hEMA_Trend_Volume"
timeframe = "4h"
leverage = 1.0