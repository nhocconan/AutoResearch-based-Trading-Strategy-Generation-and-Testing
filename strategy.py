#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA(34) trend filter and volume confirmation.
# Long when price breaks above upper Donchian band + 1d EMA up + volume spike.
# Short when price breaks below lower Donchian band + 1d EMA down + volume spike.
# Uses daily EMA for trend direction to avoid whipsaw in choppy markets.
# Volume confirmation ensures breakouts have conviction.
# Designed for fewer, high-quality trades to minimize fee drag.
name = "4h_Donchian20_1dEMA34_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d EMA trend filter
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: break above upper Donchian + 1d EMA up + volume confirmation
            if (price > highest_high[i] and price > ema_1d_aligned[i] and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below lower Donchian + 1d EMA down + volume confirmation
            elif (price < lowest_low[i] and price < ema_1d_aligned[i] and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price re-enters Donchian channel (below midpoint) or 1d EMA turns down
            midpoint = (highest_high[i] + lowest_low[i]) / 2
            if price < midpoint or price < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price re-enters Donchian channel (above midpoint) or 1d EMA turns up
            midpoint = (highest_high[i] + lowest_low[i]) / 2
            if price > midpoint or price > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals