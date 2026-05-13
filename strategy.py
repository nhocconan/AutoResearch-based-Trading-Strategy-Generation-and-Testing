#!/usr/bin/env python3
# Hypothesis: 6h Williams %R with 12h EMA20 trend filter and volume confirmation.
# Long when Williams %R crosses above -80 from oversold, price > 12h EMA20, and volume > 1.3x 20-bar average.
# Short when Williams %R crosses below -20 from overbought, price < 12h EMA20, and volume > 1.3x 20-bar average.
# Exit when Williams %R returns to neutral zone (-50) or volume drops below 0.7x average.
# Uses discrete sizing 0.25 to target 50-150 total trades over 4 years on 6h timeframe.
# Designed to capture mean reversion moves within the trend, filtering weak signals via volume and trend alignment.

name = "6h_WilliamsR_12hEMA20_Trend_VolumeConfirm"
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
    
    lookback = 14  # for Williams %R
    vol_lookback = 20  # for volume average
    
    # Calculate Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate EMA(20) on 12h close
    if len(close_12h) < 20:
        ema_20_12h = np.full(len(close_12h), np.nan)
    else:
        ema_20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 12h EMA to 6h timeframe (wait for 12h bar to close)
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=vol_lookback, min_periods=vol_lookback).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema_20_12h_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R crosses above -80 (from below), price > 12h EMA20, volume spike
            if (williams_r[i] > -80 and williams_r[i-1] <= -80 and 
                close[i] > ema_20_12h_aligned[i] and 
                volume[i] > 1.3 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R crosses below -20 (from above), price < 12h EMA20, volume spike
            elif (williams_r[i] < -20 and williams_r[i-1] >= -20 and 
                  close[i] < ema_20_12h_aligned[i] and 
                  volume[i] > 1.3 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R returns to -50 or volume drops significantly
            if (williams_r[i] >= -50 or 
                volume[i] < 0.7 * avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R returns to -50 or volume drops significantly
            if (williams_r[i] <= -50 or 
                volume[i] < 0.7 * avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals