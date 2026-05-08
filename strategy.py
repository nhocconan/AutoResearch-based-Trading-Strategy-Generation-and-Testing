#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R with 1d trend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions; fade extremes when 1d trend opposes (mean reversion),
# follow breakouts when 1d trend aligns (trend continuation). Uses 1d EMA(34) for trend and 50-period volume spike.
# Target: 60-100 total trades over 4 years (15-25/year) to minimize fee drag.

name = "4h_WilliamsR_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Williams %R (14-period)
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid div by zero
    
    # 1d EMA(34) for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_up = ema_34_1d[1:] > ema_34_1d[:-1]  # Rising EMA = uptrend
    trend_up = np.concatenate([[False], trend_up])  # Align with 1d index
    
    # Volume confirmation: 50-period volume spike (2.0x EMA)
    vol_ema = pd.Series(volume).ewm(span=50, adjust=False, min_periods=50).mean().values
    vol_confirm = volume > (vol_ema * 2.0)
    
    # Align 1d indicators to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, trend_up.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for Williams %R and volume EMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(trend_up_aligned[i]) or
            np.isnan(vol_ema[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry conditions
            # Oversold bounce in downtrend (mean reversion)
            if (trend_up_aligned[i] <= 0.5 and  # 1d downtrend
                williams_r_aligned[i] <= -80 and  # Oversold
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Breakout above resistance in uptrend (trend follow)
            elif (trend_up_aligned[i] > 0.5 and  # 1d uptrend
                  williams_r_aligned[i] >= -20 and  # Overbought breakout
                  vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short entry conditions
            # Overbought reversal in uptrend (mean reversion)
            elif (trend_up_aligned[i] > 0.5 and  # 1d uptrend
                  williams_r_aligned[i] >= -20 and  # Overbought
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
            # Oversold breakdown in downtrend (trend follow)
            elif (trend_up_aligned[i] <= 0.5 and  # 1d downtrend
                  williams_r_aligned[i] <= -80 and  # Oversold breakdown
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: reverse signal or mean reversion
            if (trend_up_aligned[i] <= 0.5 and  # 1d downtrend
                williams_r_aligned[i] >= -50):  # Return to mean
                signals[i] = 0.0
                position = 0
            elif (trend_up_aligned[i] > 0.5 and  # 1d uptrend
                  williams_r_aligned[i] <= -80):  # Re-extreme oversold
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: reverse signal or mean reversion
            if (trend_up_aligned[i] > 0.5 and  # 1d uptrend
                williams_r_aligned[i] <= -50):  # Return to mean
                signals[i] = 0.0
                position = 0
            elif (trend_up_aligned[i] <= 0.5 and  # 1d downtrend
                  williams_r_aligned[i] >= -20):  # Re-extreme overbought
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals