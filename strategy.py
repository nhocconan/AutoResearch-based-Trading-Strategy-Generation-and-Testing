#!/usr/bin/env python3
# Hypothesis: 6h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation (>1.8x 20-bar avg volume).
# Uses Donchian channel from 6h timeframe for breakout structure, 12h EMA50 for higher timeframe trend alignment, and volume spike for participation confirmation.
# Designed for BTC/ETH with discrete sizing (0.25) to minimize fee churn while capturing strong momentum moves in both bull and bear markets.
# Target: 50-150 total trades over 4 years on 6h timeframe.

name = "6h_Donchian20_12hEMA50_Volume_Confirm_v1"
timeframe = "6h"
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
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 6h Donchian channels (20-period)
    lookback_donch = 20
    highest_high = pd.Series(high).rolling(window=lookback_donch, min_periods=lookback_donch).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=lookback_donch, min_periods=lookback_donch).min().shift(1).values
    
    # Calculate average volume for confirmation (20-period)
    lookback_vol = 20
    avg_volume = pd.Series(volume).rolling(window=lookback_vol, min_periods=lookback_vol).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(max(lookback_donch, lookback_vol, 1), n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian upper channel, close > 12h EMA50, volume spike
            if (high[i] > highest_high[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume[i] > 1.8 * avg_volume[i]):
                signals[i] = 0.25  # Full position on breakout
                position = 1
            # SHORT: Price breaks below Donchian lower channel, close < 12h EMA50, volume spike
            elif (low[i] < lowest_low[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume[i] > 1.8 * avg_volume[i]):
                signals[i] = -0.25  # Full position on breakout
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # CONTINUE LONG: Reduce to half position if still above upper channel and volume OK
            if (high[i] > highest_high[i] and 
                volume[i] > avg_volume[i]):
                signals[i] = 0.125  # Half position to reduce churn
            else:
                signals[i] = 0.0  # Exit if breaks below upper channel or low volume
                position = 0
        elif position == -1:
            # CONTINUE SHORT: Reduce to half position if still below lower channel and volume OK
            if (low[i] < lowest_low[i] and 
                volume[i] > avg_volume[i]):
                signals[i] = -0.125  # Half position to reduce churn
            else:
                signals[i] = 0.0  # Exit if breaks above lower channel or low volume
                position = 0
    
    return signals