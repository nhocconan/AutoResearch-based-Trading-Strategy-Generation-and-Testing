#!/usr/bin/env python3
# Hypothesis: 1d Donchian(20) breakout with 1w EMA(34) trend filter and volume confirmation.
# Long when price breaks above Donchian upper band with price > 1w EMA34 (bullish trend) and volume > 1.5x 20-bar average.
# Short when price breaks below Donchian lower band with price < 1w EMA34 (bearish trend) and volume > 1.5x average.
# Exit when price reverses and closes below/above the Donchian middle band (mean reversion exit).
# Uses discrete position sizing 0.25. Target: 30-100 total trades over 4 years on 1d timeframe.
# EMA trend filter ensures we trade with the higher timeframe trend, avoiding counter-trend whipsaws.
# Volume confirmation validates breakout strength. Donchian exit provides clear, objective stop.
# Donchian channels work well in both trending and ranging markets, providing clear breakout levels.
# Using 1d timeframe with 1w HTF to reduce trade frequency and avoid fee drag.

name = "1d_Donchian20_1wEMA34_Trend_VolumeConfirm"
timeframe = "1d"
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
    
    lookback = 20  # for Donchian and volume average
    
    # Calculate Donchian channels
    donchian_upper = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().shift(1).values
    donchian_lower = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().shift(1).values
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA(34) on 1w close
    if len(close_1w) < 34:
        ema_34_1w = np.full(len(close_1w), np.nan)
    else:
        ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1w EMA to 1d timeframe (wait for 1w bar to close)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=lookback, min_periods=lookback).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_middle[i]) or np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian upper with bullish 1w EMA trend and volume spike
            if (close[i] > donchian_upper[i] and 
                close[i] > ema_34_1w_aligned[i] and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian lower with bearish 1w EMA trend and volume spike
            elif (close[i] < donchian_lower[i] and 
                  close[i] < ema_34_1w_aligned[i] and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below Donchian middle (mean reversion)
            if close[i] < donchian_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above Donchian middle (mean reversion)
            if close[i] > donchian_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals