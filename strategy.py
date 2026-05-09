#!/usr/bin/env python3
# Hypothesis: 6h Williams Alligator + Elder Ray combination with weekly trend filter
# Long when: Alligator bullish (jaw < teeth < lips), Elder Bull Power > 0, weekly close > weekly open
# Short when: Alligator bearish (jaw > teeth > lips), Elder Bear Power < 0, weekly close < weekly open
# Exit when: Alligator direction changes or Elder power crosses zero
# Position size: 0.25 to limit drawdown. Target: 50-150 total trades over 4 years.

name = "6h_Alligator_ElderRay_WeeklyTrend"
timeframe = "6h"
leverage = 1.0

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
    
    # Williams Alligator (13,8,5) - smoothed with SMMA
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (prev_smma * (period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(high, 13)  # Typically uses median price, but high for simplicity
    teeth = smma(high, 8)
    lips = smma(high, 5)
    
    # Elder Ray - Bull Power and Bear Power
    # Bull Power = High - EMA(13)
    # Bear Power = Low - EMA(13)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly trend: bullish if close > open, bearish if close < open
    weekly_bullish = df_1w['close'].values > df_1w['open'].values
    weekly_bearish = df_1w['close'].values < df_1w['open'].values
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish)
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Alligator bullish + Bull Power > 0 + Weekly bullish
            if (jaw[i] < teeth[i] and teeth[i] < lips[i] and  # Jaw < Teeth < Lips
                bull_power[i] > 0 and 
                weekly_bullish_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Alligator bearish + Bear Power < 0 + Weekly bearish
            elif (jaw[i] > teeth[i] and teeth[i] > lips[i] and  # Jaw > Teeth > Lips
                  bear_power[i] < 0 and 
                  weekly_bearish_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Alligator turns bearish OR Bull Power <= 0
            if not (jaw[i] < teeth[i] and teeth[i] < lips[i]) or bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Alligator turns bullish OR Bear Power >= 0
            if not (jaw[i] > teeth[i] and teeth[i] > lips[i]) or bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals