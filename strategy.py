#!/usr/bin/env python3
# Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation.
# Long when price breaks above upper Donchian channel and close > 1d EMA34 with volume > 1.8x 20-bar average.
# Short when price breaks below lower Donchian channel and close < 1d EMA34 with volume > 1.8x 20-bar average.
# Uses discrete sizing 0.25 to target 50-150 total trades over 4 years on 12h timeframe.
# Donchian structure provides objective breakout levels; 1d EMA34 filter ensures alignment with daily trend;
# volume confirmation reduces false breakouts. Works in bull markets via breakouts and in bear markets via
# mean-reversion at extreme levels (price reverts to EMA34 after overextended moves).

name = "12h_Donchian20_1dEMA34_Trend_VolumeConfirm"
timeframe = "12h"
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
    
    # Get 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=lookback, min_periods=lookback).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Calculate Donchian channels for current 12h bar using last 20 periods
        period_high = np.max(high[i-lookback+1:i+1])
        period_low = np.min(low[i-lookback+1:i+1])
        
        if position == 0:
            # LONG: Price breaks above upper Donchian, close > 1d EMA34, volume spike
            if (high[i] > period_high and 
                close[i] > ema_34_1d_aligned[i] and 
                volume[i] > 1.8 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower Donchian, close < 1d EMA34, volume spike
            elif (low[i] < period_low and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume[i] > 1.8 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below lower Donchian OR volume drops below average
            if (low[i] < period_low or 
                volume[i] < avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above upper Donchian OR volume drops below average
            if (high[i] > period_high or 
                volume[i] < avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals