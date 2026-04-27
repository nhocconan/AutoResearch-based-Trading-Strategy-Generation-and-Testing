#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray with 1d trend filter.
# Uses Alligator (3 SMAs: Jaw=13, Teeth=8, Lips=5) on 6h for trend direction.
# Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low (13-period EMA).
# Long when: Teeth > Jaw (uptrend), Bull Power > 0, and 1d close > EMA34.
# Short when: Teeth < Jaw (downtrend), Bear Power > 0, and 1d close < EMA34.
# Exit when Alligator reverses or Elder Power fails.
# Designed for ~20-30 trades/year with clear trend/filter alignment.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Williams Alligator on 6h: SMAs of median price
    median_price = (high + low) / 2.0
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values  # 13-period
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values    # 8-period
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values    # 5-period
    
    # Elder Ray Power (13-period EMA of close)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = ema13 - low   # Bear Power = EMA13 - Low
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Alligator components and Elder Power
    start_idx = 13  # max period for jaw calculation
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema34_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Alligator trend: Teeth > Jaw = uptrend, Teeth < Jaw = downtrend
        bullish_alligator = teeth[i] > jaw[i]
        bearish_alligator = teeth[i] < jaw[i]
        
        # Elder Ray confirmation
        bullish_elder = bull_power[i] > 0
        bearish_elder = bear_power[i] > 0
        
        # 1d trend filter
        bullish_1d = close > ema34_aligned[i]
        bearish_1d = close < ema34_aligned[i]
        
        if position == 0:
            # Long: uptrend + bullish elder power + bullish 1d trend
            if bullish_alligator and bullish_elder and bullish_1d:
                signals[i] = size
                position = 1
            # Short: downtrend + bearish elder power + bearish 1d trend
            elif bearish_alligator and bearish_elder and bearish_1d:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: trend turns down or elder power fails
            if not bullish_alligator or not bullish_elder:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: trend turns up or elder power fails
            if not bearish_alligator or not bearish_elder:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WilliamsAlligator_ElderRay_1dTrend"
timeframe = "6h"
leverage = 1.0