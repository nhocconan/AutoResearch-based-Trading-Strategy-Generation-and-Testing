#!/usr/bin/env python3
"""
6h_618_Retracement_With_Volume_Filter
Hypothesis: In trending markets, price often retraces to the 61.8% Fibonacci level of the recent swing before continuing. 
This strategy identifies the 6h trend using 12h EMA, waits for a pullback to the 61.8% retracement level of the last swing, 
and enters with volume confirmation. Works in both bull and bear trends by following the higher-timeframe direction.
"""

name = "6h_618_Retracement_With_Volume_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend and swing calculation (once before loop)
    df_12h = get_htf_data(prices, '12h')
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate swing points on 12h: recent swing high and low
    # We'll use the highest high and lowest low over the last 20 periods
    lookback = 20
    highest_high = pd.Series(df_12h['high']).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(df_12h['low']).rolling(window=lookback, min_periods=lookback).min().values
    
    # Align swing levels to 6h
    highest_high_aligned = align_htf_to_ltf(prices, df_12h, highest_high)
    lowest_low_aligned = align_htf_to_ltf(prices, df_12h, lowest_low)
    
    # Calculate 61.8% retracement level
    # In uptrend: retracement = highest_high - 0.618 * (highest_high - lowest_low)
    # In downtrend: retracement = lowest_low + 0.618 * (highest_high - lowest_low)
    range_12h = highest_high_aligned - lowest_low_aligned
    retracement_level = np.where(
        close > ema_50_12h_aligned,  # Uptrend condition
        highest_high_aligned - 0.618 * range_12h,
        lowest_low_aligned + 0.618 * range_12h
    )
    
    # Volume confirmation: current volume > 1.8x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_confirm = volume > (1.8 * vol_ma)
    
    # Avoid extremely low volatility periods
    price_range = pd.Series(high - low).rolling(window=12, min_periods=12).mean().values
    avg_price = pd.Series(close).rolling(window=12, min_periods=12).mean().values
    range_pct = price_range / avg_price
    volatility_filter = range_pct > 0.008  # At least 0.8% average range
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if position == 0:
            # LONG: Uptrend + price at 61.8% retracement + volume confirmation
            if (close[i] > ema_50_12h_aligned[i] and  # Uptrend
                abs(close[i] - retracement_level[i]) < (0.003 * close[i]) and  # Near 61.8% level (0.3% tolerance)
                volume_confirm[i] and 
                volatility_filter[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Downtrend + price at 61.8% retracement + volume confirmation
            elif (close[i] < ema_50_12h_aligned[i] and  # Downtrend
                  abs(close[i] - retracement_level[i]) < (0.003 * close[i]) and  # Near 61.8% level
                  volume_confirm[i] and 
                  volatility_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Trend reversal or price moves significantly against us
            if close[i] < ema_50_12h_aligned[i] or close[i] > highest_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend reversal or price moves significantly against us
            if close[i] > ema_50_12h_aligned[i] or close[i] < lowest_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals