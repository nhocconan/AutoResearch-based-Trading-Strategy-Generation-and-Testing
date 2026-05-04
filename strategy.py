#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray convergence with 1d trend filter
# Long when: price > Alligator Jaw (teeth > lips) AND Bull Power > 0 AND 1d close > EMA50
# Short when: price < Alligator Jaw (teeth < lips) AND Bear Power < 0 AND 1d close < EMA50
# Williams Alligator uses SMAs of median price: Jaw=13, Teeth=8, Lips=5 (all shifted)
# Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
# 1d EMA50 filter ensures we only trade in the direction of the higher timeframe trend
# Target: 50-150 total trades over 4 years = 12-37/year. Discrete size: 0.25

name = "6h_WilliamsAlligator_ElderRay_1dTrend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data for HTF trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_bullish_1d = close_1d > ema_50_1d
    trend_bearish_1d = close_1d < ema_50_1d
    
    # Align 1d trend to 6h timeframe
    trend_bullish_aligned = align_htf_to_ltf(prices, df_1d, trend_bullish_1d.astype(float))
    trend_bearish_aligned = align_htf_to_ltf(prices, df_1d, trend_bearish_1d.astype(float))
    
    # Calculate Williams Alligator components on 6h data
    median_price = (high + low) / 2.0
    
    # Alligator Jaw (Blue) - 13-period SMA, shifted 8 bars
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values
    # Alligator Teeth (Red) - 8-period SMA, shifted 5 bars
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values
    # Alligator Lips (Green) - 5-period SMA, shifted 3 bars
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Calculate Elder Ray components on 6h data
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = low - ema_13   # Bear Power = Low - EMA13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period for all indicators
        # Skip if any value is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(trend_bullish_aligned[i]) or np.isnan(trend_bearish_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: 
            # 1. Price > Alligator Jaw (teeth > lips indicates bullish alignment)
            # 2. Bull Power > 0 (strong buying pressure)
            # 3. 1d bullish trend
            if (close[i] > jaw[i] and 
                teeth[i] > lips[i] and 
                bull_power[i] > 0 and 
                trend_bullish_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short conditions:
            # 1. Price < Alligator Jaw (teeth < lips indicates bearish alignment)
            # 2. Bear Power < 0 (strong selling pressure)
            # 3. 1d bearish trend
            elif (close[i] < jaw[i] and 
                  teeth[i] < lips[i] and 
                  bear_power[i] < 0 and 
                  trend_bearish_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below Jaw OR Alligator alignment turns bearish
            if (close[i] < jaw[i] or 
                teeth[i] < lips[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Jaw OR Alligator alignment turns bullish
            if (close[i] > jaw[i] or 
                teeth[i] > lips[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals