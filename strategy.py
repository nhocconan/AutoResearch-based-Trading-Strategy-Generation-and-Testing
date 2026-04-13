#!/usr/bin/env python3
"""
6h_1w_1d_Price_Channel_Retest_With_Volume_Confirmation
Hypothesis: On 6h timeframe, price tends to retest prior weekly highs/lows (from 1w) and daily support/resistance (from 1d) before continuing trend. Enter on retest with volume confirmation.
- Long when: price retraces to weekly low + bounces (close > open) + 6h volume > 1.5x 20-period average
- Short when: price retraces to weekly high + rejects (close < open) + 6h volume > 1.5x 20-period average
- Use daily high/loose as additional filters: avoid longs near daily high, avoid shorts near daily low
- Exit on opposite retest or volume drought
Designed for 6h to capture swing points in both bull/bear markets with low trade frequency.
"""

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
    open_ = prices['open'].values
    volume = prices['volume'].values
    
    # Get weekly data for key levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly high and low (from completed weekly candles)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    
    # Align weekly levels to 6h (they update only on weekly close)
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    # Get daily data for filter levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Align daily levels to 6h
    daily_high_aligned = align_htf_to_ltf(prices, df_1d, daily_high)
    daily_low_aligned = align_htf_to_ltf(prices, df_1d, daily_low)
    daily_close_aligned = align_htf_to_ltf(prices, df_1d, daily_close)
    
    # 6h volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_condition = volume > (vol_ma_20 * 1.5)
    
    # Price action filters
    bullish_candle = close > open_
    bearish_candle = close < open_
    
    # Proximity to weekly levels (within 0.5%)
    proximity_to_weekly_low = (low <= weekly_low_aligned * 1.005) & (low >= weekly_low_aligned * 0.995)
    proximity_to_weekly_high = (high >= weekly_high_aligned * 0.995) & (high <= weekly_high_aligned * 1.005)
    
    # Avoid trading near daily extremes
    near_daily_high = (high >= daily_high_aligned * 0.995)  # within 0.5% of daily high
    near_daily_low = (low <= daily_low_aligned * 1.005)    # within 0.5% of daily low
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position
    
    for i in range(100, n):
        # Skip if weekly/daily data not ready
        if (np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or
            np.isnan(daily_high_aligned[i]) or np.isnan(daily_low_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long setup: retest of weekly low with bounce
        long_setup = (proximity_to_weekly_low[i] and 
                     bullish_candle[i] and 
                     vol_condition[i] and 
                     not near_daily_high[i])  # avoid buying near daily high
        
        # Short setup: retest of weekly high with rejection
        short_setup = (proximity_to_weekly_high[i] and 
                      bearish_candle[i] and 
                      vol_condition[i] and 
                      not near_daily_low[i])  # avoid selling near daily low
        
        # Exit conditions: opposite retest or volume drought
        vol_drought = volume[i] < (vol_ma_20[i] * 0.5)  # volume < 50% of average
        
        if position == 0:
            if long_setup:
                position = 1
                signals[i] = position_size
            elif short_setup:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: retest of weekly high or volume drought
            if proximity_to_weekly_high[i] or vol_drought:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: retest of weekly low or volume drought
            if proximity_to_weekly_low[i] or vol_drought:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1w_1d_Price_Channel_Retest_With_Volume_Confirmation"
timeframe = "6h"
leverage = 1.0