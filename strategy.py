#!/usr/bin/env python3
# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation.
# Long when price breaks above upper Donchian channel and close > 1w EMA34 with volume > 1.5x 20-bar average.
# Short when price breaks below lower Donchian channel and close < 1w EMA34 with volume > 1.5x 20-bar average.
# Uses discrete sizing 0.25 to target 30-100 total trades over 4 years on 1d timeframe.
# Donchian channels provide structure; combined with weekly trend and volume spike filters.
# Works in bull markets via breakouts and in bear markets via mean-reversion at extreme levels.

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
    
    lookback = 20  # for Donchian channels and volume average
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Donchian channels (20-period)
    upper_channel = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().shift(1).values
    lower_channel = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().shift(1).values
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=lookback, min_periods=lookback).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above upper Donchian, close > 1w EMA34, volume spike
            if (high[i] > upper_channel[i] and 
                close[i] > ema_34_1w_aligned[i] and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower Donchian, close < 1w EMA34, volume spike
            elif (low[i] < lower_channel[i] and 
                  close[i] < ema_34_1w_aligned[i] and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below lower Donchian OR volume dries up (< 0.8x average)
            if (low[i] < lower_channel[i] or 
                volume[i] < 0.8 * avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above upper Donchian OR volume dries up (< 0.8x average)
            if (high[i] > upper_channel[i] or 
                volume[i] < 0.8 * avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals