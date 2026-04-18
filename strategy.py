#!/usr/bin/env python3
"""
6h_OrderBlock_Bounce_v1
Strategy: 6h Order Block bounce with weekly trend filter.
Long: Price bounces from bullish order block (bullish candle before downtrend) in weekly uptrend.
Short: Price rejects from bearish order block (bearish candle before uptrend) in weekly downtrend.
Uses weekly order blocks identified from prior candle's close vs open.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for order blocks and trend
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly OHLC
    weekly_open = df_1w['open'].values
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_volume = df_1w['volume'].values
    
    # Identify order blocks: bullish (close > open) and bearish (close < open)
    bullish_ob = weekly_close > weekly_open
    bearish_ob = weekly_close < weekly_open
    
    # Order block levels: use the body of the candle
    ob_top = np.maximum(weekly_open, weekly_close)  # higher of open/close
    ob_bottom = np.minimum(weekly_open, weekly_close)  # lower of open/close
    
    # Weekly trend: EMA50 vs EMA200
    weekly_ema_50 = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_ema_200 = pd.Series(weekly_close).ewm(span=200, adjust=False, min_periods=200).mean().values
    weekly_uptrend = weekly_ema_50 > weekly_ema_200
    weekly_downtrend = weekly_ema_50 < weekly_ema_200
    
    # Align weekly data to 6h timeframe
    ob_top_aligned = align_htf_to_ltf(prices, df_1w, ob_top)
    ob_bottom_aligned = align_htf_to_ltf(prices, df_1w, ob_bottom)
    bullish_ob_aligned = align_htf_to_ltf(prices, df_1w, bullish_ob.astype(float))
    bearish_ob_aligned = align_htf_to_ltf(prices, df_1w, bearish_ob.astype(float))
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for weekly EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ob_top_aligned[i]) or np.isnan(ob_bottom_aligned[i]) or
            np.isnan(bullish_ob_aligned[i]) or np.isnan(bearish_ob_aligned[i]) or
            np.isnan(weekly_uptrend_aligned[i]) or np.isnan(weekly_downtrend_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Order block bounce conditions
        # Long: price touches or enters bullish OB and weekly uptrend
        long_condition = (bullish_ob_aligned[i] > 0.5 and 
                         low[i] <= ob_top_aligned[i] and 
                         high[i] >= ob_bottom_aligned[i] and
                         weekly_uptrend_aligned[i] > 0.5)
        
        # Short: price touches or enters bearish OB and weekly downtrend
        short_condition = (bearish_ob_aligned[i] > 0.5 and
                          high[i] >= ob_bottom_aligned[i] and
                          low[i] <= ob_top_aligned[i] and
                          weekly_downtrend_aligned[i] > 0.5)
        
        if position == 0:
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below OB bottom or trend change
            if low[i] < ob_bottom_aligned[i] or weekly_uptrend_aligned[i] < 0.5:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above OB top or trend change
            if high[i] > ob_top_aligned[i] or weekly_downtrend_aligned[i] < 0.5:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_OrderBlock_Bounce_v1"
timeframe = "6h"
leverage = 1.0