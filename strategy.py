#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with 1w EMA20 trend filter and volume confirmation.
# Williams %R measures overbought/oversold levels. Long when %R < -80 (oversold) in uptrend (price > 1w EMA20).
# Short when %R > -20 (overbought) in downtrend (price < 1w EMA20).
# Volume filter ensures participation (volume > 1.5x 20-period average).
# Designed for 12h timeframe with low trade frequency (target: 15-25/year) to avoid fee drag.
# Uses 1w EMA20 for trend filter to avoid counter-trend trades in strong trends.
# Williams %R is effective in ranging markets and captures reversals in trends.
name = "12h_WilliamsR_1wEMA20_VolumeFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams %R (14 period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # 1w EMA20 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(williams_r[i]) or np.isnan(ema20_1w_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R < -80 (oversold), price > 1w EMA20 (uptrend), volume filter
            long_cond = (williams_r[i] < -80) and (close[i] > ema20_1w_aligned[i]) and volume_filter[i]
            # Short conditions: Williams %R > -20 (overbought), price < 1w EMA20 (downtrend), volume filter
            short_cond = (williams_r[i] > -20) and (close[i] < ema20_1w_aligned[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R > -20 (overbought) OR price < 1w EMA20 (trend change)
            if williams_r[i] > -20 or close[i] < ema20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R < -80 (oversold) OR price > 1w EMA20 (trend change)
            if williams_r[i] < -80 or close[i] > ema20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals