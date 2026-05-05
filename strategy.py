#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray combination with 1w trend filter
# Long when: Alligator jaws < teeth < lips (bullish alignment) AND Bull Power > 0 AND price > 1w EMA50
# Short when: Alligator jaws > teeth > lips (bearish alignment) AND Bear Power < 0 AND price < 1w EMA50
# Exit when: Alligator alignment breaks (jaws > teeth or teeth > lips) OR power crosses zero
# Uses 6h primary timeframe with 1w HTF for trend filter and 1d for Elder Ray calculation
# Williams Alligator identifies trend phases, Elder Ray measures bull/bear power behind moves
# Discrete sizing (0.25) to limit fee drag and manage drawdown in ranging markets
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe

name = "6h_Alligator_ElderRay_1wEMA50"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Calculate 1w EMA50 trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get 1d data ONCE before loop for Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 13-period EMA for Elder Ray (standard setting)
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bull_power_1d = high_1d - ema_13_1d
    bear_power_1d = low_1d - ema_13_1d
    
    # Align 1d Elder Ray to 6h timeframe
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Get 6h data ONCE before loop for Williams Alligator
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 40:
        return np.zeros(n)
    
    # Calculate Williams Alligator lines (standard periods: 13,8,5 with shifts 8,5,3)
    # Jaws: Blue line - 13-period SMMA smoothed 8 bars ahead
    # Teeth: Red line - 8-period SMMA smoothed 5 bars ahead  
    # Lips: Green line - 5-period SMMA smoothed 3 bars ahead
    median_6h = (high + low) / 2  # Williams Alligator uses median price
    
    # Calculate SMMA (Smoothed Moving Average) - equivalent to RMA/Wilder's smoothing
    def smma(source, period):
        if len(source) < period:
            return np.full_like(source, np.nan)
        result = np.full_like(source, np.nan)
        # First value is simple SMA
        result[period-1] = np.mean(source[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
        for i in range(period, len(source)):
            result[i] = (result[i-1] * (period-1) + source[i]) / period
        return result
    
    smma_13 = smma(median_6h, 13)
    smma_8 = smma(median_6h, 8)
    smma_5 = smma(median_6h, 5)
    
    # Apply the shifts (jaws shifted 8, teeth shifted 5, lips shifted 3)
    jaws = np.full_like(smma_13, np.nan)
    teeth = np.full_like(smma_8, np.nan)
    lips = np.full_like(smma_5, np.nan)
    
    if len(smma_13) > 8:
        jaws[8:] = smma_13[:-8]
    if len(smma_8) > 5:
        teeth[5:] = smma_8[:-5]
    if len(smma_5) > 3:
        lips[3:] = smma_5[:-3]
    
    # Align Alligator lines to 6h timeframe (same df_6h)
    jaws_aligned = align_htf_to_ltf(prices, df_6h, jaws)
    teeth_aligned = align_htf_to_ltf(prices, df_6h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_6h, lips)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(jaws_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power_1d_aligned[i]) or np.isnan(bear_power_1d_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Bullish Alligator alignment AND Bull Power > 0 AND above 1w EMA50
            if (jaws_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < lips_aligned[i] and
                bull_power_1d_aligned[i] > 0 and close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Bearish Alligator alignment AND Bear Power < 0 AND below 1w EMA50
            elif (jaws_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > lips_aligned[i] and
                  bear_power_1d_aligned[i] < 0 and close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator alignment breaks OR Bull Power <= 0
            if (jaws_aligned[i] >= teeth_aligned[i] or teeth_aligned[i] >= lips_aligned[i] or
                bull_power_1d_aligned[i] <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator alignment breaks OR Bear Power >= 0
            if (jaws_aligned[i] <= teeth_aligned[i] or teeth_aligned[i] <= lips_aligned[i] or
                bear_power_1d_aligned[i] >= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals