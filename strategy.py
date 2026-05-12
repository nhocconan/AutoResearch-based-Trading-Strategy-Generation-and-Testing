#!/usr/bin/env python3
# 4H_DONCHIAN_BREAKOUT_20_VOLUME_CONFIRMATION_12H_TREND_FILTER
# Hypothesis: Donchian(20) breakouts on 4h with volume confirmation and 12h trend filter.
# Works in bull markets (breakouts continuation) and bear markets (reversals at extremes).
# Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years).

name = "4H_DONCHIAN_BREAKOUT_20_VOLUME_CONFIRMATION_12H_TREND_FILTER"
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
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # EMA50 for trend filter
    ema50 = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_12h, ema50)
    
    # Average volume for confirmation
    avg_vol = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any critical data is not ready
        if np.isnan(ema50_aligned[i]) or np.isnan(avg_vol[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Calculate Donchian channels
        donchian_high = np.max(high[i-19:i+1])
        donchian_low = np.min(low[i-19:i+1])
        
        if position == 0:
            # LONG: Price breaks above Donchian high with volume confirmation and uptrend
            if (close[i] > donchian_high and 
                volume[i] > 1.5 * avg_vol[i] and 
                close[i] > ema50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low with volume confirmation and downtrend
            elif (close[i] < donchian_low and 
                  volume[i] > 1.5 * avg_vol[i] and 
                  close[i] < ema50_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls below Donchian low or trend reversal
            if (close[i] < donchian_low or 
                close[i] <= ema50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises above Donchian high or trend reversal
            if (close[i] > donchian_high or 
                close[i] >= ema50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals