#!/usr/bin/env python3
"""
6h Williams Alligator + Elder Ray with 1d trend filter
Hypothesis: Alligator identifies trend direction (jaw-teeth-lips alignment), Elder Ray measures bull/bear power behind the move.
Combined with 1d trend filter to avoid counter-trend trades. Works in trending markets (both bull and bear) by only taking
trades in direction of 1d trend. Uses Williams Alligator (SMAs with offsets) and Elder Ray (EMA13-based bull/bear power).
Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_alligator_elder_ray_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams Alligator: SMAs with specific periods and offsets
    # Jaw: SMA(13, offset=8)
    # Teeth: SMA(8, offset=5) 
    # Lips: SMA(5, offset=3)
    def sma_with_offset(arr, period, offset):
        """Calculate SMA then shift forward by offset bars"""
        sma = np.full(n, np.nan)
        if n >= period:
            for i in range(period - 1, n):
                sma[i] = np.mean(arr[i - period + 1:i + 1])
        # Shift forward by offset (add offset to index)
        shifted = np.full(n, np.nan)
        for i in range(n - offset):
            shifted[i + offset] = sma[i]
        return shifted
    
    jaw = sma_with_offset(close, 13, 8)   # Blue line
    teeth = sma_with_offset(close, 8, 5)   # Red line  
    lips = sma_with_offset(close, 5, 3)    # Green line
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = np.full(n, np.nan)
    if n >= 13:
        ema13[12] = np.mean(close[:13])
        for i in range(13, n):
            ema13[i] = (close[i] * 2 + ema13[i-1] * 11) / 12
    
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Get 1d data for trend filter (close > EMA50 = bullish)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # EMA50 on 1d close
    ema_50 = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_50[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50[i] = (close_1d[i] * 2 + ema_50[i-1] * 48) / 49
    
    # 1d trend: 1 = bullish (close > EMA50), -1 = bearish (close < EMA50)
    trend_1d = np.where(close_1d > ema_50, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_exit = 0
    
    # Warmup: need enough data for Alligator components
    start = 30
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema13[i]) or np.isnan(trend_1d_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_exit += 1
            continue
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: Alligator lines cross (jaws below teeth) OR bear power strong OR against 1d trend
            if (jaw[i] < teeth[i] or    # Alligator sleeping - trend weakening
                bear_power[i] < -1.5 * np.std(bull_power[max(0, i-20):i+1]) or  # Strong bear power
                trend_1d_aligned[i] == -1):  # Against 1d trend
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                signals[i] = 0.25
            bars_since_exit += 1
        elif position == -1:  # short position
            # Exit: Alligator lines cross (jaws above teeth) OR bull power strong OR against 1d trend
            if (jaw[i] > teeth[i] or    # Alligator sleeping - trend weakening
                bull_power[i] > 1.5 * np.std(bear_power[max(0, i-20):i+1]) or  # Strong bull power
                trend_1d_aligned[i] == 1):  # Against 1d trend
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                signals[i] = -0.25
            bars_since_exit += 1
        else:
            # Look for entries with minimum bars between trades
            if bars_since_exit >= 6:  # Minimum 6 bars between trades
                # Alligator conditions: lips > teeth > jaw (bullish) OR lips < teeth < jaw (bearish)
                bullish_alligator = lips[i] > teeth[i] and teeth[i] > jaw[i]
                bearish_alligator = lips[i] < teeth[i] and teeth[i] < jaw[i]
                
                # Elder Ray conditions: bull power > 0 AND increasing, OR bear power < 0 AND decreasing
                bull_power_strong = bull_power[i] > 0 and (i == start or bull_power[i] > bull_power[i-1])
                bear_power_strong = bear_power[i] < 0 and (i == start or bear_power[i] < bear_power[i-1])
                
                # Long: bullish Alligator + strong bull power + bullish 1d trend
                if bullish_alligator and bull_power_strong and trend_1d_aligned[i] == 1:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_exit = 0
                # Short: bearish Alligator + strong bear power + bearish 1d trend
                elif bearish_alligator and bear_power_strong and trend_1d_aligned[i] == -1:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                    bars_since_exit = 0
                else:
                    signals[i] = 0.0
                    bars_since_exit += 1
            else:
                signals[i] = 0.0
                bars_since_exit += 1
    
    return signals