#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + Williams Alligator with 1d trend filter
# Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures bull/bear strength
# Williams Alligator (Jaw=EMA13, Teeth=EMA8, Lips=EMA5) identifies trend direction and strength
# Combined: Long when Bull Power > 0 AND Lips > Teeth > Jaw (bullish alignment) AND 1d EMA50 uptrend
# Short when Bear Power > 0 AND Jaw > Teeth > Lips (bearish alignment) AND 1d EMA50 downtrend
# Uses 6h timeframe for signal generation with discrete position sizing (0.25) to minimize fee churn
# Target: 50-150 total trades over 4 years = 12-37/year for 6h timeframe
# Works in bull markets via Elder Ray strength + Alligator alignment, in bear via same logic for shorts

name = "6h_ElderRay_Alligator_1dEMA50_Trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Pre-compute session hours (08-20 UTC) - index is DatetimeIndex
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 6h data ONCE before loop for Elder Ray and Alligator
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 50:
        return np.zeros(n)
    
    # Calculate 6h EMA indicators for Elder Ray and Alligator
    close_6h = df_6h['close'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # EMA5 for Alligator Lips
    ema5_6h = pd.Series(close_6h).ewm(span=5, adjust=False, min_periods=5).mean().values
    # EMA8 for Alligator Teeth
    ema8_6h = pd.Series(close_6h).ewm(span=8, adjust=False, min_periods=8).mean().values
    # EMA13 for Alligator Jaw and Elder Ray
    ema13_6h = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Align 6h indicators to lower timeframe
    ema5_6h_aligned = align_htf_to_ltf(prices, df_6h, ema5_6h)
    ema8_6h_aligned = align_htf_to_ltf(prices, df_6h, ema8_6h)
    ema13_6h_aligned = align_htf_to_ltf(prices, df_6h, ema13_6h)
    high_6h_aligned = align_htf_to_ltf(prices, df_6h, high_6h)
    low_6h_aligned = align_htf_to_ltf(prices, df_6h, low_6h)
    
    # Calculate Elder Ray components
    bull_power = high_6h_aligned - ema13_6h_aligned  # High - EMA13
    bear_power = ema13_6h_aligned - low_6h_aligned   # EMA13 - Low
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Check for NaN values in indicators
        if (np.isnan(ema5_6h_aligned[i]) or np.isnan(ema8_6h_aligned[i]) or 
            np.isnan(ema13_6h_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: Bull Power > 0 AND Lips > Teeth > Jaw (bullish Alligator) AND 1d Uptrend
            if (bull_power[i] > 0 and 
                ema5_6h_aligned[i] > ema8_6h_aligned[i] and 
                ema8_6h_aligned[i] > ema13_6h_aligned[i] and
                ema50_1d_aligned[i] > ema50_1d_aligned[max(0, i-1)]):  # 1d EMA50 rising
                signals[i] = 0.25
                position = 1
            # Short conditions: Bear Power > 0 AND Jaw > Teeth > Lips (bearish Alligator) AND 1d Downtrend
            elif (bear_power[i] > 0 and 
                  ema13_6h_aligned[i] > ema8_6h_aligned[i] and 
                  ema8_6h_aligned[i] > ema5_6h_aligned[i] and
                  ema50_1d_aligned[i] < ema50_1d_aligned[max(0, i-1)]):  # 1d EMA50 falling
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Bear Power > 0 OR Alligator loses bullish alignment OR 1d trend turns down
            if (bear_power[i] > 0 or 
                not (ema5_6h_aligned[i] > ema8_6h_aligned[i] > ema13_6h_aligned[i]) or
                ema50_1d_aligned[i] < ema50_1d_aligned[max(0, i-1)]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Bull Power > 0 OR Alligator loses bearish alignment OR 1d trend turns up
            if (bull_power[i] > 0 or 
                not (ema13_6h_aligned[i] > ema8_6h_aligned[i] > ema5_6h_aligned[i]) or
                ema50_1d_aligned[i] > ema50_1d_aligned[max(0, i-1)]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals