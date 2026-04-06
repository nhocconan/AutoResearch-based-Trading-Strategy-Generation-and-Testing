#!/usr/bin/env python3
"""
1d Volume-Weighted Price Action + Weekly Trend Filter
Hypothesis: On daily timeframe, price respects weekly trend when accompanied by volume confirmation.
Long when price > VWAP(20) AND weekly EMA(8) rising AND volume > 1.5x average.
Short when price < VWAP(20) AND weekly EMA(8) falling AND volume > 1.5x average.
Uses VWAP for intraday value area and weekly EMA for trend filter to avoid counter-trend trades.
Targets 50-100 trades over 4 years (12-25/year) to minimize fee drag.
Works in bull markets (follow weekly uptrend) and bear markets (short weekly downtrend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14330_1d_vwap_weekly_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for trend filter (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    close_weekly = df_weekly['close'].values
    
    # Calculate 8-period EMA for weekly trend
    ema_weekly = pd.Series(close_weekly).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema_weekly_rising = ema_weekly > np.roll(ema_weekly, 1)
    ema_weekly_falling = ema_weekly < np.roll(ema_weekly, 1)
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    ema_weekly_rising_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly_rising.astype(float))
    ema_weekly_falling_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly_falling.astype(float))
    
    # Daily data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # VWAP (20-period)
    typical_price = (high + low + close) / 3.0
    vwap_numerator = typical_price * volume
    vwap_numerator_sum = pd.Series(vwap_numerator).rolling(window=20, min_periods=20).sum().values
    volume_sum = pd.Series(volume).rolling(window=20, min_periods=20).sum().values
    vwap = vwap_numerator_sum / (volume_sum + 1e-10)
    
    # Volume filter: require volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 20
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(vwap[i]) or np.isnan(ema_weekly_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: reverse signal or price crosses VWAP
        if position == 1:  # long position
            if (ema_weekly_falling_aligned[i] == 1.0) or (close[i] < vwap[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if (ema_weekly_rising_aligned[i] == 1.0) or (close[i] > vwap[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: price vs VWAP + weekly trend + volume filter
            long_setup = (close[i] > vwap[i]) and (ema_weekly_rising_aligned[i] == 1.0) and vol_filter[i]
            short_setup = (close[i] < vwap[i]) and (ema_weekly_falling_aligned[i] == 1.0) and vol_filter[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_setup:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals