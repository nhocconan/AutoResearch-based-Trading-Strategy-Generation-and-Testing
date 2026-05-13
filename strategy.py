#!/usr/bin/env python3
# Hypothesis: 12h Williams %R mean reversion with 1d EMA50 trend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions; EMA50 ensures alignment with daily trend;
# Volume confirmation filters low-participation signals. Designed for fewer, higher-quality trades
# on 12h timeframe to minimize fee drag while working in both bull and bear markets.

name = "12h_WilliamsR_MeanReversion_1dEMA50_VolumeConfirm"
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
    
    # Calculate Williams %R (14-period) from prior candle only
    lookback_wr = 14
    highest_high = pd.Series(high).rolling(window=lookback_wr, min_periods=lookback_wr).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=lookback_wr, min_periods=lookback_wr).min().shift(1).values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate average volume for confirmation (20-period)
    lookback_vol = 20
    avg_volume = pd.Series(volume).rolling(window=lookback_vol, min_periods=lookback_vol).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(max(lookback_wr, lookback_vol, 1), n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(avg_volume[i]) or
            np.isnan(williams_r[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R oversold (< -80), price above 1d EMA50, volume spike
            if (williams_r[i] < -80 and 
                close[i] > ema_50_1d_aligned[i] and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R overbought (> -20), price below 1d EMA50, volume spike
            elif (williams_r[i] > -20 and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R rises above -50 (mean reversion) OR volume drops below average
            if (williams_r[i] > -50 or 
                volume[i] < avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R falls below -50 (mean reversion) OR volume drops below average
            if (williams_r[i] < -50 or 
                volume[i] < avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals