#!/usr/bin/env python3
# Hypothesis: 6h Williams %R mean reversion with 12h EMA20 trend filter and volume confirmation (>1.3x 20-bar avg volume).
# Williams %R identifies overbought/oversold conditions; 12h EMA20 ensures higher timeframe trend alignment;
# Volume confirmation filters low-participation signals. Designed for 50-150 total trades over 4 years on 6h timeframe.
# Works in both bull and bear markets by fading extremes in the direction of the 12h trend.

name = "6h_WilliamsR_MeanReversion_12hEMA20_VolumeConfirm_v1"
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
    
    # Calculate 12h EMA20 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    ema_20_12h = pd.Series(df_12h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # Calculate Williams %R (14-period) - using prior candle only
    lookback_wr = 14
    highest_high = pd.Series(high).rolling(window=lookback_wr, min_periods=lookback_wr).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=lookback_wr, min_periods=lookback_wr).min().shift(1).values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Calculate average volume for confirmation (20-period)
    lookback_vol = 20
    avg_volume = pd.Series(volume).rolling(window=lookback_vol, min_periods=lookback_vol).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(max(lookback_wr, lookback_vol, 1), n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema_20_12h_aligned[i]) or np.isnan(williams_r[i]) or
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R oversold (< -80) and price > 12h EMA20 and volume spike
            if (williams_r[i] < -80 and 
                close[i] > ema_20_12h_aligned[i] and 
                volume[i] > 1.3 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R overbought (> -20) and price < 12h EMA20 and volume spike
            elif (williams_r[i] > -20 and 
                  close[i] < ema_20_12h_aligned[i] and 
                  volume[i] > 1.3 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R returns above -50 OR volume drops below average
            if (williams_r[i] > -50 or 
                volume[i] < avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R returns below -50 OR volume drops below average
            if (williams_r[i] < -50 or 
                volume[i] < avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals