#!/usr/bin/env python3
name = "4h_Donchian_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0

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
    
    # Donchian(20) on primary timeframe
    period = 20
    highest = np.full(n, np.nan)
    lowest = np.full(n, np.nan)
    for i in range(period, n):
        highest[i] = np.max(high[i-period:i])
        lowest[i] = np.min(low[i-period:i])
    
    # 12h trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_surge = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(period, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest[i]) or 
            np.isnan(lowest[i]) or 
            np.isnan(ema34_12h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian band + 12h uptrend + volume surge
            if close[i] > highest[i] and close[i-1] <= highest[i] and ema34_12h_aligned[i] > close[i] * 0.99 and vol_surge[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian band + 12h downtrend + volume surge
            elif close[i] < lowest[i] and close[i-1] >= lowest[i] and ema34_12h_aligned[i] < close[i] * 1.01 and vol_surge[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price closes below lower Donchian band OR trend turns down
            if close[i] < lowest[i] or ema34_12h_aligned[i] < close[i] * 0.99:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price closes above upper Donchian band OR trend turns up
            if close[i] > highest[i] or ema34_12h_aligned[i] > close[i] * 1.01:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Donchian(20) breakout with 12h EMA34 trend filter and volume surge confirmation
# Captures strong momentum moves in both bull and bear markets.
# Long when price breaks above 20-period high with 12h uptrend and volume confirmation.
# Short when price breaks below 20-period low with 12h downtrend and volume confirmation.
# Exits when price reverses back into the channel or trend changes.
# Volume surge filter reduces false breakouts. Position size 0.25 manages risk.
# Expected trade count: 20-40/year per symbol, well within limits.