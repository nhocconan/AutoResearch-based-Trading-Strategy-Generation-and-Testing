#!/usr/bin/env python3
# 6h_12h_1d_LW_VolBreakout_1dTrend
# Hypothesis: 6h Larry Williams Volatility Breakout with 1d trend filter.
# Long when price > open + 0.5 * (prev day range) in uptrend.
# Short when price < open - 0.5 * (prev day range) in downtrend.
# Uses 12h/1d for trend and volatility context. Targets 50-150 trades over 4 years.

name = "6h_12h_1d_LW_VolBreakout_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data for volatility and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily range and breakout levels
    # Larry Williams Volatility Breakout:
    # Long breakout: open + k * (prev day high - low)
    # Short breakdown: open - k * (prev day high - low)
    # Using k=0.5 for balanced sensitivity
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_range = prev_high - prev_low
    prev_range[0] = np.nan  # First day has no previous
    
    # Breakout levels for each day
    buy_level = open_ + 0.5 * prev_range
    sell_level = open_ - 0.5 * prev_range
    
    # Align levels to 6h timeframe (wait for 1d bar to close)
    buy_level_aligned = align_htf_to_ltf(prices, df_1d, buy_level)
    sell_level_aligned = align_htf_to_ltf(prices, df_1d, sell_level)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d close for trend direction
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Volume confirmation: 2x 4-period average (approx 1 day of 6h bars)
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 1d data + EMA34 + vol MA
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(buy_level_aligned[i]) or
            np.isnan(sell_level_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend: 1d close > EMA34
        uptrend = close_1d_aligned[i] > ema_34_1d_aligned[i]
        downtrend = close_1d_aligned[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation (2x average for significance)
        volume_surge = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Breakout above buy level in uptrend with volume surge
            if close[i] > buy_level_aligned[i] and uptrend and volume_surge:
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below sell level in downtrend with volume surge
            elif close[i] < sell_level_aligned[i] and downtrend and volume_surge:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Long exit: close below buy level or trend fails
                if close[i] < buy_level_aligned[i] or not uptrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short exit: close above sell level or trend fails
                if close[i] > sell_level_aligned[i] or not downtrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals