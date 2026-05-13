#!/usr/bin/env python3
# Hypothesis: 6h Williams %R extreme + 12h EMA50 trend filter + volume spike.
# Long when Williams %R < -80 (oversold) AND 12h EMA50 rising AND volume > 2.0x average.
# Short when Williams %R > -20 (overbought) AND 12h EMA50 falling AND volume > 2.0x average.
# Williams %R identifies exhaustion points in ranging/bear markets, EMA50 filters trend direction,
# volume spike confirms participation. Designed for low frequency (12-30 trades/year) to avoid fee drag.
# Works in bull (buy oversold dips in uptrend) and bear (sell overbought rallies in downtrend).

name = "6h_WilliamsR_Extreme_12hEMA50_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R(14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r[highest_high == lowest_low] = -50.0
    
    # Calculate average volume for confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA(50) on 12h data
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 6h timeframe (wait for 12h bar to close)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R < -80 (oversold) AND 12h EMA50 rising AND volume > 2.0x average
            if (williams_r[i] < -80.0 and 
                ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1] and 
                volume[i] > 2.0 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R > -20 (overbought) AND 12h EMA50 falling AND volume > 2.0x average
            elif (williams_r[i] > -20.0 and 
                  ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1] and 
                  volume[i] > 2.0 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R > -50 (momentum fading) OR EMA50 turns down
            if williams_r[i] > -50.0 or ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R < -50 (momentum fading) OR EMA50 turns up
            if williams_r[i] < -50.0 or ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals