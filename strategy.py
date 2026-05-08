#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R with 1d EMA trend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions: long when %R < -80 (oversold) and price > 1d EMA50 (uptrend),
# short when %R > -20 (overbought) and price < 1d EMA50 (downtrend).
# Volume confirmation requires 1d volume > 1.2x 20-day average to ensure institutional participation.
# Designed to work in both bull and bear markets by following the higher timeframe trend.
# Target: 80-150 total trades over 4 years (20-38/year) to avoid excessive fee drag.

name = "4h_WilliamsR_1dEMA50_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Daily data for EMA and volume
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 2:
        return np.zeros(n)
    
    # Williams %R(14) on 4h data
    williams_period = 14
    highest_high = pd.Series(high).rolling(window=williams_period, min_periods=williams_period).max().values
    lowest_low = pd.Series(low).rolling(window=williams_period, min_periods=williams_period).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r[highest_high == lowest_low] = -50  # Avoid division by zero
    
    # Daily EMA50 for trend direction
    close_d = df_d['close'].values
    ema50_d = pd.Series(close_d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Daily volume filter: current volume > 1.2x 20-period average
    volume_d = df_d['volume'].values
    vol_ma20_d = pd.Series(volume_d).rolling(window=20, min_periods=20).mean().values
    volume_filter_d = volume_d > (1.2 * vol_ma20_d)
    
    # Align daily indicators to 4h timeframe
    ema50_aligned = align_htf_to_ltf(prices, df_d, ema50_d)
    volume_filter = align_htf_to_ltf(prices, df_d, volume_filter_d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(williams_period, 50)  # Sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema50_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R oversold (< -80) and price above daily EMA50 (uptrend) with volume
            long_cond = (williams_r[i] < -80) and (close[i] > ema50_aligned[i]) and volume_filter[i]
            # Short conditions: Williams %R overbought (> -20) and price below daily EMA50 (downtrend) with volume
            short_cond = (williams_r[i] > -20) and (close[i] < ema50_aligned[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R returns above -50 (momentum fading) or price breaks below daily EMA50
            if williams_r[i] > -50 or close[i] < ema50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R returns below -50 (momentum fading) or price breaks above daily EMA50
            if williams_r[i] < -50 or close[i] > ema50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals