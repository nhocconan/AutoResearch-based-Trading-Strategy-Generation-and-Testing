#!/usr/bin/env python3
# Hypothesis: 1d Williams %R with 1w trend filter and volume confirmation.
# Williams %R measures overbought/oversold levels: long when %R < -80 (oversold) AND price > 1w EMA50 (uptrend) AND volume > 1.5x average.
# Short when %R > -20 (overbought) AND price < 1w EMA50 (downtrend) AND volume > 1.5x average.
# Exit when %R reverts to midpoint (-50) or trend reverses.
# Uses 1d timeframe for lower frequency, Williams %R for mean reversion, 1w EMA for trend filter, volume for confirmation.
# Target: 30-100 total trades over 4 years (7-25/year). Works in bull via buying dips in uptrend, bear via selling rallies in downtrend.

name = "1d_WilliamsR_1wTrend_Volume_v1"
timeframe = "1d"
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
    
    # Get 1d data for Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Williams %R(14) on 1d data
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume filter: current 1d volume > 1.5x 20-period average
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_filter_1d = volume_1d > (1.5 * vol_ma_1d)
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA(50) on 1w close for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema50_1w_aligned[i]) or
            np.isnan(vol_ma_1d[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R < -80 (oversold) AND price > 1w EMA50 (uptrend) AND volume confirmation
            if williams_r[i] < -80 and close[i] > ema50_1w_aligned[i] and volume_filter_1d[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R > -20 (overbought) AND price < 1w EMA50 (downtrend) AND volume confirmation
            elif williams_r[i] > -20 and close[i] < ema50_1w_aligned[i] and volume_filter_1d[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R >= -50 (reverting to midpoint) OR trend reversal (price < 1w EMA50)
            if williams_r[i] >= -50 or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R <= -50 (reverting to midpoint) OR trend reversal (price > 1w EMA50)
            if williams_r[i] <= -50 or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals