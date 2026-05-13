#!/usr/bin/env python3
# Hypothesis: 12h Donchian(20) breakout with 1w EMA200 trend filter and volume spike confirmation.
# Long when price breaks above upper Donchian channel with 1w EMA200 uptrend and volume > 2.0x average.
# Short when price breaks below lower Donchian channel with 1w EMA200 downtrend and volume > 2.0x average.
# Uses discrete sizing 0.25. Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe.
# Donchian channels provide robust structure, 1w EMA200 ensures we trade with the long-term trend.
# Volume spike confirms institutional participation. Works in bull markets via upward breaks and in bear markets via downward breaks.

name = "12h_Donchian20_1wEMA200_VolumeSpike_v1"
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
    
    # Calculate Donchian channel (20-period)
    lookback = 20
    if n < lookback + 1:
        return np.zeros(n)
    
    upper_channel = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().shift(1).values
    lower_channel = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().shift(1).values
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    
    # Get 1w data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA200 on 1w data
    ema_200_1w = pd.Series(close_1w).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Align 1w EMA200 to 12h timeframe (wait for 1w bar to close)
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback + 20, n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(ema_200_1w_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above upper Donchian with 1w EMA200 uptrend and volume spike
            if (close[i] > upper_channel[i] and 
                close[i] > ema_200_1w_aligned[i] and 
                volume[i] > 2.0 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower Donchian with 1w EMA200 downtrend and volume spike
            elif (close[i] < lower_channel[i] and 
                  close[i] < ema_200_1w_aligned[i] and 
                  volume[i] > 2.0 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below lower Donchian (reversal signal)
            if close[i] < lower_channel[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above upper Donchian (reversal signal)
            if close[i] > upper_channel[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals