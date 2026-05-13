#!/usr/bin/env python3
# Hypothesis: 12h Williams %R with 1w trend filter and volume confirmation.
# Williams %R measures overbought/oversold levels: Long when %R < -80 (oversold) AND price > 1w EMA50 (uptrend) AND volume > 1.3x average
# Short when %R > -20 (overbought) AND price < 1w EMA50 (downtrend) AND volume > 1.3x average
# Exit when %R reverses to midpoint (-50) OR trend fails
# Uses 12h timeframe for lower frequency, Williams %R for mean reversion in extremes, 1w EMA for trend filter, volume for confirmation.
# Target: 50-150 total trades over 4 years (12-37/year). Works in bull via buying dips in uptrend, bear via selling rallies in downtrend.

name = "12h_WilliamsR_1wTrend_Volume_v1"
timeframe = "12h"
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
    
    # Get 12h data for Williams %R calculation
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate Williams %R(14) on 12h: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_12h) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume filter: current 12h volume > 1.3x 20-period average
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_filter_12h = volume_12h > (1.3 * vol_ma_12h)
    
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
            np.isnan(vol_ma_12h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R < -80 (oversold) AND price > 1w EMA50 (uptrend) AND volume confirmation
            if williams_r[i] < -80 and close[i] > ema50_1w_aligned[i] and volume_filter_12h[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R > -20 (overbought) AND price < 1w EMA50 (downtrend) AND volume confirmation
            elif williams_r[i] > -20 and close[i] < ema50_1w_aligned[i] and volume_filter_12h[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R > -50 (reverting from oversold) OR trend fails (price < 1w EMA50)
            if williams_r[i] > -50 or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R < -50 (reverting from overbought) OR trend fails (price > 1w EMA50)
            if williams_r[i] < -50 or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals