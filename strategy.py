#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with 1d Elder Ray (bull/bear power) and 1w trend filter.
# Bull Power = High - EMA13(Close), Bear Power = EMA13(Close) - Low
# Long: Bull Power > 0 and Bear Power < 0 (bullish imbalance) + price > weekly EMA34
# Short: Bear Power > 0 and Bull Power < 0 (bearish imbalance) + price < weekly EMA34
# Uses 1d for Elder Ray calculation (captures intra-day strength/weakness),
# 1w EMA34 for trend filter to avoid counter-trend trades.
# Works in bull markets (trend-following with strength confirmation)
# and bear markets (avoids longs in downtrend, shorts in uptrend).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 1d data for Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA13 on daily close for Elder Ray
    close_1d_series = pd.Series(close_1d)
    ema13_1d = close_1d_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high_1d - ema13_1d  # High - EMA13
    bear_power = ema13_1d - low_1d   # EMA13 - Low
    
    # 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    ema34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        bull = bull_power_aligned[i]
        bear = bear_power_aligned[i]
        weekly_ema = ema34_1w_aligned[i]
        
        if position == 0:
            # Long: bullish imbalance (bull>0, bear<0) + price above weekly EMA
            if (bull > 0 and bear < 0 and price > weekly_ema):
                position = 1
                signals[i] = position_size
            # Short: bearish imbalance (bear>0, bull<0) + price below weekly EMA
            elif (bear > 0 and bull < 0 and price < weekly_ema):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: bullish imbalance breaks or price crosses below weekly EMA
            if (bull <= 0 or bear >= 0 or price < weekly_ema):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: bearish imbalance breaks or price crosses above weekly EMA
            if (bear <= 0 or bull >= 0 or price > weekly_ema):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_1w_Elder_Ray_Weekly_Trend"
timeframe = "6h"
leverage = 1.0