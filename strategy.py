#!/usr/bin/env python3
"""
Hypothesis: Daily Donchian(20) breakout with weekly EMA trend filter and volume confirmation.
Long when price breaks above Donchian high and weekly EMA > previous weekly EMA.
Short when price breaks below Donchian low and weekly EMA < previous weekly EMA.
Exit when price returns to Donchian midline or trend reverses.
Works in trending markets by capturing breakouts; avoids whipsaws with weekly trend filter.
Volume confirmation ensures institutional participation in breakouts.
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
    
    # Load weekly data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA(20) for trend direction
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_20_1w_prev = np.roll(ema_20_1w, 1)
    ema_20_1w_prev[0] = ema_20_1w[0]
    ema_rising = ema_20_1w > ema_20_1w_prev
    ema_falling = ema_20_1w < ema_20_1w_prev
    ema_rising_aligned = align_htf_to_ltf(prices, df_1w, ema_rising)
    ema_falling_aligned = align_htf_to_ltf(prices, df_1w, ema_falling)
    
    # Daily Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2.0
    
    # Daily volume filter - average volume over 20 periods
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high, weekly EMA rising, volume above average
            if (close[i] > donch_high[i] and 
                ema_rising_aligned[i] and 
                volume[i] > avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low, weekly EMA falling, volume above average
            elif (close[i] < donch_low[i] and 
                  ema_falling_aligned[i] and 
                  volume[i] > avg_volume[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price returns to Donchian midline or trend turns down
                if (close[i] <= donch_mid[i] or not ema_rising_aligned[i]):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price returns to Donchian midline or trend turns up
                if (close[i] >= donch_mid[i] or not ema_falling_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Donchian20_1wEMA20_Trend_Volume"
timeframe = "1d"
leverage = 1.0