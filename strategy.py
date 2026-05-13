#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and 1w EMA50 trend filter.
# Long when price breaks above 20-bar high with 1d volume > 1.5x 20-bar average and 1w EMA50 rising.
# Short when price breaks below 20-bar low with 1d volume > 1.5x 20-bar average and 1w EMA50 falling.
# Exit when price crosses the 20-bar EMA10 in opposite direction.
# Uses discrete position sizing 0.25. Target: 75-200 total trades over 4 years on 4h timeframe.
# 1w EMA50 ensures we trade with the higher timeframe trend, reducing false breakouts in ranging markets.
# Volume confirmation ensures breakouts have conviction.

name = "4h_Donchian20_1wEMA50_Trend_VolumeConfirm_v1"
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
    
    # Calculate Donchian channels (20-period)
    lookback = 20
    if n < lookback + 1:
        return np.zeros(n)
    
    # Rolling high/low for Donchian channels
    high_roll = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().shift(1).values
    low_roll = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().shift(1).values
    
    # Calculate 20-bar EMA for exit
    ema10 = pd.Series(close).ewm(span=10, min_periods=10, adjust=False).mean().values
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 on 1w data
    ema50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align 1w EMA50 to 4h timeframe (wait for 1w bar to close)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback + 20, n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(ema10[i]) or np.isnan(avg_volume[i]) or 
            np.isnan(ema50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Check if 1w EMA50 is rising/falling (trend filter)
            ema50_rising = ema50_1w_aligned[i] > ema50_1w_aligned[i-1]
            ema50_falling = ema50_1w_aligned[i] < ema50_1w_aligned[i-1]
            
            # LONG: Price breaks above Donchian high with volume confirmation and rising 1w EMA50
            if (close[i] > high_roll[i] and 
                volume[i] > 1.5 * avg_volume[i] and 
                ema50_rising):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low with volume confirmation and falling 1w EMA50
            elif (close[i] < low_roll[i] and 
                  volume[i] > 1.5 * avg_volume[i] and 
                  ema50_falling):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below 20-bar EMA10
            if close[i] < ema10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above 20-bar EMA10
            if close[i] > ema10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals