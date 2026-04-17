#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h volume confirmation and 12h EMA34 trend filter.
Long when price breaks above upper Donchian(20) with volume > 1.3x 12h average volume AND 12h EMA34 rising.
Short when price breaks below lower Donchian(20) with volume > 1.3x 12h average volume AND 12h EMA34 falling.
Exit when price touches the opposite Donchian level or when trend reverses (EMA34 cross).
Uses 12h for volume and trend confirmation, 4h for entry/exit and Donchian calculation.
Designed to capture medium-term trends with volume confirmation to avoid false breakouts.
Target: 20-40 trades/year per symbol (80-160 total over 4 years).
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
    
    # Get 12h data for volume MA and EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    volume_12h = df_12h['volume'].values
    close_12h = df_12h['close'].values
    
    # Volume confirmation: 1.3x 12h average volume
    vol_ma_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    
    # Trend filter: 12h EMA34
    ema_34_12h = pd.Series(close_12h).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_rising = ema_34_12h > np.roll(ema_34_12h, 1)
    ema_34_falling = ema_34_12h < np.roll(ema_34_12h, 1)
    ema_34_rising[0] = False
    ema_34_falling[0] = False
    ema_34_rising_aligned = align_htf_to_ltf(prices, df_12h, ema_34_rising)
    ema_34_falling_aligned = align_htf_to_ltf(prices, df_12h, ema_34_falling)
    
    # 4h Donchian channels (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup period
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(vol_ma_20_aligned[i]) or 
            np.isnan(ema_34_rising_aligned[i]) or
            np.isnan(ema_34_falling_aligned[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 12h average volume
        volume_confirmed = volume[i] > 1.3 * vol_ma_20_aligned[i]
        
        # Donchian levels
        upper_channel = highest_high[i]
        lower_channel = lowest_low[i]
        
        # Breakout conditions
        breakout_upper = close[i] > upper_channel
        breakout_lower = close[i] < lower_channel
        
        if position == 0:
            # Long: break above upper Donchian with volume confirmation and rising 12h EMA
            if (breakout_upper and volume_confirmed and ema_34_rising_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below lower Donchian with volume confirmation and falling 12h EMA
            elif (breakout_lower and volume_confirmed and ema_34_falling_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price touches lower Donchian OR trend turns bearish
            if (close[i] <= lower_channel or 
                (ema_34_falling_aligned[i] and not ema_34_rising_aligned[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price touches upper Donchian OR trend turns bullish
            if (close[i] >= upper_channel or 
                (ema_34_rising_aligned[i] and not ema_34_falling_aligned[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hVolume_12hEMA34_Trend"
timeframe = "4h"
leverage = 1.0