#!/usr/bin/env python3
"""
12h_1d_1w_Chaikin_Oscillator_With_Trend_Filter_v1
12-hour strategy using Chaikin Oscillator (3,10) with 1-day trend filter.
Long when Chaikin > 0 and 1d EMA20 > EMA50.
Short when Chaikin < 0 and 1d EMA20 < EMA50.
Uses volume accumulation/distribution to detect institutional accumulation.
Designed for low trade frequency (<30/year) to minimize fee drag.
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
    volume = prices['volume'].values
    
    # === 1-day EMA for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === Chaikin Oscillator (3,10) on price data ===
    # Money Flow Multiplier = [(Close - Low) - (High - Close)] / (High - Low)
    # Avoid division by zero
    hl_range = high - low
    mfm = np.where(hl_range != 0, ((close - low) - (high - close)) / hl_range, 0.0)
    # Money Flow Volume = MFM * Volume
    mfv = mfm * volume
    # ADL = cumulative sum of MFV
    adl = np.cumsum(mfv)
    # Chaikin Oscillator = EMA(3, ADL) - EMA(10, ADL)
    adl_series = pd.Series(adl)
    ema3_adl = adl_series.ewm(span=3, adjust=False, min_periods=3).mean().values
    ema10_adl = adl_series.ewm(span=10, adjust=False, min_periods=10).mean().values
    chaikin = ema3_adl - ema10_adl
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema20_1d_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(chaikin[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Trend filter: 1d EMA20 > EMA50 for uptrend, < for downtrend
        uptrend = ema20_1d_aligned[i] > ema50_1d_aligned[i]
        downtrend = ema20_1d_aligned[i] < ema50_1d_aligned[i]
        
        # Chaikin Oscillator signals
        chaikin_positive = chaikin[i] > 0
        chaikin_negative = chaikin[i] < 0
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: positive Chaikin + uptrend
            if chaikin_positive and uptrend:
                signals[i] = 0.25
                position = 1
                continue
            # Short: negative Chaikin + downtrend
            elif chaikin_negative and downtrend:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: reverse signal or trend change
        elif position == 1:
            # Exit long: Chaikin turns negative OR trend turns down
            if chaikin_negative or downtrend:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Chaikin turns positive OR trend turns up
            if chaikin_positive or uptrend:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_1w_Chaikin_Oscillator_With_Trend_Filter_v1"
timeframe = "12h"
leverage = 1.0