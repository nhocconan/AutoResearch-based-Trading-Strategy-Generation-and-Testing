#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R (14) extreme reversal with 1w EMA34 trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions. In strong trends (1w EMA34), 
# extreme readings often precede continuations rather than reversals. 
# Long: %R < -80 (oversold) AND price > 1w EMA34 AND volume spike
# Short: %R > -20 (overbought) AND price < 1w EMA34 AND volume spike
# Uses discrete position sizing (0.25) to minimize fee churn. Designed for 12-25 trades/year.
# Works in bull markets via buying oversold dips in uptrends and in bear markets via selling overbought rallies in downtrends.

name = "6h_WilliamsR14_1wEMA34_VolumeSpike_TrendFilter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA34 trend filter from prior completed 1w bar
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_shifted = np.roll(ema34_1w, 1)
    ema34_1w_shifted[0] = np.nan
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w_shifted)
    
    # Calculate Williams %R (14) on 6h timeframe
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high_14 - close) / (highest_high_14 - lowest_low_14) * -100
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # Volume confirmation: 20-period EMA of volume on 6h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(williams_r[i]) or
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R oversold (< -80) AND above 1w EMA34 AND volume spike
            if williams_r[i] < -80 and close[i] > ema34_1w_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R overbought (> -20) AND below 1w EMA34 AND volume spike
            elif williams_r[i] > -20 and close[i] < ema34_1w_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R rises above -50 (momentum weakening) OR price breaks below 1w EMA34
            if williams_r[i] > -50 or close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R falls below -50 (momentum weakening) OR price breaks above 1w EMA34
            if williams_r[i] < -50 or close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals