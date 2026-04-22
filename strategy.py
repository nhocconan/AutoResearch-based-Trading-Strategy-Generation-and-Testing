#!/usr/bin/env python3
"""
Hypothesis: 1-day 20-period Donchian breakout with 1-week trend filter and volume confirmation.
Long when price breaks above upper band, 1-week EMA50 is rising, and 1-day volume > 20-period average.
Short when price breaks below lower band, 1-week EMA50 is falling, and 1-day volume > 20-period average.
Exit when price crosses the opposite band or volume drops below average.
Donchian channels capture breakouts in trending markets; EMA50 filter avoids counter-trend trades.
Volume confirmation ensures institutional participation. Works in bull markets via breakouts and in bear
markets via short breakdowns. Weekly trend filter reduces whipsaws during sideways periods.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1-week data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_rising = ema_50_1w > np.roll(ema_50_1w, 1)
    ema_50_1w_rising[0] = False
    ema_50_1w_falling = ema_50_1w < np.roll(ema_50_1w, 1)
    ema_50_1w_falling[0] = False
    
    ema_50_1w_rising_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w_rising)
    ema_50_1w_falling_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w_falling)
    
    # Load 1-day data for Donchian and volume - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-period Donchian channels
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period average volume
    avg_vol_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align to lower timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    avg_vol_20_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or
            np.isnan(avg_vol_20_aligned[i]) or np.isnan(ema_50_1w_rising_aligned[i]) or
            np.isnan(ema_50_1w_falling_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above upper band, weekly EMA rising, volume above average
            if (high[i] > upper_20_aligned[i] and 
                ema_50_1w_rising_aligned[i] and 
                volume_1d[i] > avg_vol_20_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower band, weekly EMA falling, volume above average
            elif (low[i] < lower_20_aligned[i] and 
                  ema_50_1w_falling_aligned[i] and 
                  volume_1d[i] > avg_vol_20_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price falls below lower band
                if low[i] < lower_20_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price rises above upper band
                if high[i] > upper_20_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Donchian20_1wEMA50_Trend_Volume"
timeframe = "1d"
leverage = 1.0