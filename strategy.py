#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
# Long when price breaks above 4h Donchian upper channel and close > 1d EMA50 with volume > 1.8x 20-bar average.
# Short when price breaks below 4h Donchian lower channel and close < 1d EMA50 with volume > 1.8x 20-bar average.
# Uses discrete sizing 0.25 to target 75-150 total trades over 4 years on 4h timeframe.
# Donchian channels provide robust structure; 1d EMA50 filters counter-trend noise; volume confirms momentum.
# Designed for fewer, higher-quality trades to avoid fee drag while working in both bull and bear markets.

name = "4h_Donchian20_1dEMA50_Trend_VolumeConfirm"
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
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 4h Donchian channels (20-period)
    lookback_donch = 20
    upper_channel = pd.Series(high).rolling(window=lookback_donch, min_periods=lookback_donch).max().shift(1).values
    lower_channel = pd.Series(low).rolling(window=lookback_donch, min_periods=lookback_donch).min().shift(1).values
    
    # Calculate average volume for confirmation (20-period)
    lookback_vol = 20
    avg_volume = pd.Series(volume).rolling(window=lookback_vol, min_periods=lookback_vol).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(max(lookback_donch, lookback_vol), n):
        # Skip if any required data is NaN
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above upper Donchian, close > 1d EMA50, volume spike
            if (high[i] > upper_channel[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume[i] > 1.8 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower Donchian, close < 1d EMA50, volume spike
            elif (low[i] < lower_channel[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume[i] > 1.8 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below lower Donchian OR volume drops below average
            if (low[i] < lower_channel[i] or 
                volume[i] < avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above upper Donchian OR volume drops below average
            if (high[i] > upper_channel[i] or 
                volume[i] < avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals