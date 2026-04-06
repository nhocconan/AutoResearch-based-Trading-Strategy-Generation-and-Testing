#!/usr/bin/env python3
"""
12h Camarilla Pivot Reversal with Volume Spike and Choppiness Filter
Hypothesis: Camarilla pivot levels from 1-day provide strong intraday support/resistance.
Price rejecting these levels with volume spikes indicates reversal. Choppiness filter ensures
we only trade in trending markets (CHOP < 38.2) to avoid false signals in ranging markets.
Designed for 50-150 trades over 4 years on 12h timeframe to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_reversal_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Camarilla pivots and Choppiness (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels for 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels (based on previous day)
    pivot = (high_1d[:-1] + low_1d[:-1] + close_1d[:-1]) / 3
    range_ = high_1d[:-1] - low_1d[:-1]
    
    # Resistance levels
    r1 = close_1d[:-1] + range_ * 1.1 / 12
    r2 = close_1d[:-1] + range_ * 1.1 / 6
    r3 = close_1d[:-1] + range_ * 1.1 / 4
    r4 = close_1d[:-1] + range_ * 1.1 / 2
    
    # Support levels
    s1 = close_1d[:-1] - range_ * 1.1 / 12
    s2 = close_1d[:-1] - range_ * 1.1 / 6
    s3 = close_1d[:-1] - range_ * 1.1 / 4
    s4 = close_1d[:-1] - range_ * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe (using previous day's levels)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Choppiness Index on 1d (using high/low/close)
    def calculate_chop(high, low, close, period=14):
        """Calculate Choppiness Index"""
        atr = np.zeros_like(high)
        for i in range(1, len(high)):
            tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            atr[i] = tr
        
        # Sum of true ranges over period
        tr_sum = np.zeros_like(close)
        for i in range(period, len(high)):
            tr_sum[i] = np.sum(atr[i-period+1:i+1])
        
        # Highest high and lowest low over period
        hh = np.zeros_like(high)
        ll = np.zeros_like(high)
        for i in range(period-1, len(high)):
            hh[i] = np.max(high[i-period+1:i+1])
            ll[i] = np.min(low[i-period+1:i+1])
        
        # Choppiness formula
        chop = np.full_like(close, 50.0)  # Default to neutral
        for i in range(period, len(high)):
            if hh[i] > ll[i]:  # Avoid division by zero
                chop[i] = 100 * np.log10(tr_sum[i] / (hh[i] - ll[i])) / np.log10(period)
        return chop
    
    chop = calculate_chop(high_1d, low_1d, close_1d, 14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # 12h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(20, 14)  # For volume MA and chop
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: price moves against position beyond pivot or chop increases
        if position == 1:  # long position
            # Exit: price drops below S1 or chop > 61.8 (ranging market)
            if close[i] < s1_aligned[i] or chop_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price rises above R1 or chop > 61.8
            if close[i] > r1_aligned[i] or chop_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: price rejection at Camarilla levels + volume + trending market
            # Long setup: price near S3/S4 and bouncing up
            near_s3 = abs(close[i] - s3_aligned[i]) < (r1_aligned[i] - s1_aligned[i]) * 0.15
            near_s4 = abs(close[i] - s4_aligned[i]) < (r1_aligned[i] - s1_aligned[i]) * 0.15
            bullish_rejection = close[i] > open[i] and close[i] > s1_aligned[i]
            
            # Short setup: price near R3/R4 and bouncing down
            near_r3 = abs(close[i] - r3_aligned[i]) < (r1_aligned[i] - s1_aligned[i]) * 0.15
            near_r4 = abs(close[i] - r4_aligned[i]) < (r1_aligned[i] - s1_aligned[i]) * 0.15
            bearish_rejection = close[i] < open[i] and close[i] < r1_aligned[i]
            
            volume_filter = volume[i] > vol_ma[i] * 1.8
            trending_filter = chop_aligned[i] < 38.2  # Trending market
            
            if (near_s3 or near_s4) and bullish_rejection and volume_filter and trending_filter:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif (near_r3 or near_r4) and bearish_rejection and volume_filter and trending_filter:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals