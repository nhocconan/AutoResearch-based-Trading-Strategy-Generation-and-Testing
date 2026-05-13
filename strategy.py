#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 12h EMA34 trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high with 12h EMA34 up and volume > 1.5x 20-bar average.
# Short when price breaks below Donchian(20) low with 12h EMA34 down and volume > 1.5x 20-bar average.
# Uses discrete sizing 0.25 to limit fee churn. Designed for low trade frequency (<50/year) and strong performance in both bull and bear markets via trend filter.

name = "4h_Donchian20_12hEMA34_VolumeConfirm"
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
    
    lookback = 20  # for Donchian channels and volume average
    
    # Calculate Donchian channels (20-period high/low)
    highest = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate EMA(34) on 12h close
    if len(close_12h) < 34:
        ema_34_12h = np.full(len(close_12h), np.nan)
    else:
        ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 12h EMA to 4h timeframe (wait for 12h bar to close)
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=lookback, min_periods=lookback).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(highest[i]) or np.isnan(lowest[i]) or 
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian high with bullish 12h trend and volume spike
            if (close[i] > highest[i] and 
                ema_34_12h_aligned[i] > ema_34_12h_aligned[i-1] and  # 12h EMA rising
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low with bearish 12h trend and volume spike
            elif (close[i] < lowest[i] and 
                  ema_34_12h_aligned[i] < ema_34_12h_aligned[i-1] and  # 12h EMA falling
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian low OR 12h EMA turns down
            if (close[i] < lowest[i] or 
                ema_34_12h_aligned[i] < ema_34_12h_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian high OR 12h EMA turns up
            if (close[i] > highest[i] or 
                ema_34_12h_aligned[i] > ema_34_12h_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals