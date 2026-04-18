#!/usr/bin/env python3
"""
1d_1w_Trend_Following_With_Volume_and_Weekly_Pullback
Hypothesis: Use weekly trend (price above/below weekly EMA20) as primary filter, combined with daily price action pulling back to daily EMA20 with volume confirmation. This captures trend continuation after pullbacks in both bull and bear markets. Weekly trend filter reduces whipsaws, while daily EMA20 pullback provides precise entry. Targets 15-25 trades/year by requiring alignment of weekly trend, daily EMA20 touch/penetration, and volume > 1.3x 20-day average. Works in bull markets by buying dips in uptrend, and in bear markets by selling rallies in downtrend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA20 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema20_1w = close_1w_series.ewm(span=20, adjust=False, min_periods=20).values
    
    # Align weekly EMA20 to daily timeframe
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Get daily EMA20 for pullback entries
    close_series = pd.Series(close)
    ema20_daily = close_series.ewm(span=20, adjust=False, min_periods=20).values
    
    # Volume confirmation: current volume > 1.3 x 20-day average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema20_1w_aligned[i]) or np.isnan(ema20_daily[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: weekly uptrend (price > weekly EMA20) AND price touches/below daily EMA20 with volume
            if (close[i] > ema20_1w_aligned[i] and low[i] <= ema20_daily[i] and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: weekly downtrend (price < weekly EMA20) AND price touches/above daily EMA20 with volume
            elif (close[i] < ema20_1w_aligned[i] and high[i] >= ema20_daily[i] and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: weekly trend turns down OR price breaks above weekly EMA20 by too much (extended)
            if (close[i] < ema20_1w_aligned[i] or 
                close[i] > ema20_1w_aligned[i] * 1.05):  # 5% above weekly EMA20 = extended
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: weekly trend turns up OR price breaks below weekly EMA20 by too much (extended)
            if (close[i] > ema20_1w_aligned[i] or 
                close[i] < ema20_1w_aligned[i] * 0.95):  # 5% below weekly EMA20 = extended
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_Trend_Following_With_Volume_and_Weekly_Pullback"
timeframe = "1d"
leverage = 1.0