#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d trend filter.
# Long when 6h Williams %R < -80 (oversold) and 1d close > 1d EMA50 (uptrend).
# Short when 6h Williams %R > -20 (overbought) and 1d close < 1d EMA50 (downtrend).
# Uses Williams %R for mean reversion entries and daily EMA for trend filter to avoid counter-trend trades.
# Designed for 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# Works in bull markets via long signals in uptrend and bear markets via short signals in downtrend.
name = "6h_WilliamsR_1dEMA50_Trend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_up_1d = close_1d > ema_50_1d  # True for uptrend
    trend_down_1d = close_1d < ema_50_1d  # True for downtrend
    trend_up_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_up_1d)
    trend_down_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_down_1d)
    
    # 6h Williams %R (14-period)
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) > 0, williams_r, -50.0)  # avoid division by zero
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Sufficient warmup for Williams %R and EMA
    
    for i in range(start_idx, n):
        if (np.isnan(williams_r[i]) or 
            np.isnan(trend_up_1d_aligned[i]) or np.isnan(trend_down_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long condition: Williams %R oversold AND 1d uptrend
            long_condition = (williams_r[i] < -80) and trend_up_1d_aligned[i]
            # Short condition: Williams %R overbought AND 1d downtrend
            short_condition = (williams_r[i] > -20) and trend_down_1d_aligned[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Williams %R returns to neutral (> -50) or trend turns down
            if (williams_r[i] > -50) or (not trend_up_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Williams %R returns to neutral (< -50) or trend turns up
            if (williams_r[i] < -50) or (not trend_down_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals