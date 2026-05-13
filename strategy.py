#!/usr/bin/env python3
# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation (>1.8x 30-bar avg volume).
# Uses Donchian channel from 12h for breakout signals, 1d EMA50 for higher timeframe trend alignment, volume spike for participation confirmation.
# Designed for BTC/ETH with discrete sizing (0.25) to minimize fee churn while capturing strong momentum moves in both bull and bear markets.
# Target: 50-150 total trades over 4 years on 12h timeframe.

name = "12h_Donchian20_1dEMA50_Volume_v1"
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
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 12h Donchian channel (20-period)
    lookback_donch = 20
    highest_high = pd.Series(high).rolling(window=lookback_donch, min_periods=lookback_donch).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=lookback_donch, min_periods=lookback_donch).min().shift(1).values
    
    # Calculate average volume for confirmation (30-period)
    lookback_vol = 30
    avg_volume = pd.Series(volume).rolling(window=lookback_vol, min_periods=lookback_vol).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(max(lookback_donch, lookback_vol, 1), n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian upper, close > 1d EMA50, volume spike
            if (high[i] > highest_high[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume[i] > 1.8 * avg_volume[i]):
                signals[i] = 0.25  # Full position on breakout
                position = 1
            # SHORT: Price breaks below Donchian lower, close < 1d EMA50, volume spike
            elif (low[i] < lowest_low[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume[i] > 1.8 * avg_volume[i]):
                signals[i] = -0.25  # Full position on breakout
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # CONTINUE LONG: Exit if breaks below Donchian lower or volume drops
            if (low[i] < lowest_low[i] or 
                volume[i] < 0.9 * avg_volume[i]):
                signals[i] = 0.0  # Exit
                position = 0
            else:
                signals[i] = 0.25  # Maintain position
        elif position == -1:
            # CONTINUE SHORT: Exit if breaks above Donchian upper or volume drops
            if (high[i] > highest_high[i] or 
                volume[i] < 0.9 * avg_volume[i]):
                signals[i] = 0.0  # Exit
                position = 0
            else:
                signals[i] = -0.25  # Maintain position
    
    return signals