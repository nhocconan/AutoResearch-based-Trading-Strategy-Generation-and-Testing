#!/usr/bin/env python3
# Hypothesis: 6h Williams %R mean reversion with 12h trend filter and volume spike confirmation.
# Long when Williams %R < -80 (oversold) with 12h EMA200 uptrend and volume > 2.0x average.
# Short when Williams %R > -20 (overbought) with 12h EMA200 downtrend and volume > 2.0x average.
# Uses discrete sizing 0.25. Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe.
# Williams %R identifies extreme reversals. 12h EMA200 ensures we trade with the long-term trend.
# Volume spike confirms participation. Works in bull markets via oversold bounces and in bear markets via overbought rejections.

name = "6h_WilliamsR_MeanReversion_12hEMA200_VolumeSpike_v1"
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
    
    # Calculate Williams %R (14-period)
    lookback = 14
    if n < lookback + 1:
        return np.zeros(n)
    
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 12h data for EMA200 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA200 on 12h data
    ema_200_12h = pd.Series(close_12h).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Align 12h EMA200 to 6h timeframe (wait for 12h bar to close)
    ema_200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_200_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback + 20, n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or 
            np.isnan(ema_200_12h_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R oversold (< -80) with 12h EMA200 uptrend and volume spike
            if (williams_r[i] < -80 and 
                close[i] > ema_200_12h_aligned[i] and 
                volume[i] > 2.0 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R overbought (> -20) with 12h EMA200 downtrend and volume spike
            elif (williams_r[i] > -20 and 
                  close[i] < ema_200_12h_aligned[i] and 
                  volume[i] > 2.0 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R returns above -50 (mean reversion)
            if williams_r[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R returns below -50 (mean reversion)
            if williams_r[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals