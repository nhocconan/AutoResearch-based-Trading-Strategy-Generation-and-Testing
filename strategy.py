#!/usr/bin/env python3
"""
1d_WickReversal_WeeklyTrend_Filter
Hypothesis: Uses daily long wicks (pin bars) as reversal signals in the direction of the weekly trend. 
Long when: weekly close > weekly open (bullish week) AND daily close > daily open AND lower wick > 2x body.
Short when: weekly close < weekly open (bearish week) AND daily close < daily open AND upper wick > 2x body.
Exits on opposite weekly trend change or opposite wick signal. 
Designed to capture mean-reversion within weekly trends, working in both bull and bear markets.
Target: 15-30 trades/year via strict wick and trend alignment.
"""

name = "1d_WickReversal_WeeklyTrend_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtd_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Daily OHLC
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    weekly_open = df_1w['open'].values
    weekly_close = df_1w['close'].values
    weekly_bullish = weekly_close > weekly_open
    weekly_bearish = weekly_close < weekly_open
    
    # Align weekly trend to daily
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    
    # Daily wick calculations
    body = np.abs(close - open_)
    lower_wick = np.minimum(open_, close) - low
    upper_wick = high - np.maximum(open_, close)
    
    # Avoid division by zero
    body_safe = np.where(body == 0, 1e-10, body)
    
    # Long wick conditions: wick > 2x body
    long_wick_signal = lower_wick > 2 * body
    short_wick_signal = upper_wick > 2 * body
    
    # Only consider bullish/bearish daily candles
    bullish_daily = close > open_
    bearish_daily = close < open_
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i]) or
            np.isnan(long_wick_signal[i]) or np.isnan(short_wick_signal[i]) or
            np.isnan(bullish_daily[i]) or np.isnan(bearish_daily[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: weekly bullish trend + bullish daily + long lower wick
            if (weekly_bullish_aligned[i] > 0.5 and 
                bullish_daily[i] and 
                long_wick_signal[i]):
                signals[i] = 0.25
                position = 1
            # Short: weekly bearish trend + bearish daily + long upper wick
            elif (weekly_bearish_aligned[i] > 0.5 and 
                  bearish_daily[i] and 
                  short_wick_signal[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: weekly trend turns bearish OR opposite wick signal
                if (weekly_bearish_aligned[i] > 0.5) or short_wick_signal[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: weekly trend turns bullish OR opposite wick signal
                if (weekly_bullish_aligned[i] > 0.5) or long_wick_signal[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals