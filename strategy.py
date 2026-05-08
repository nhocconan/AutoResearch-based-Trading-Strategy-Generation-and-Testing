#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d Elder Ray and volume confirmation.
# Long when: Alligator jaws (13-period SMMA) < teeth (8-period SMMA) < lips (5-period SMMA) AND 
#            Bull Power > 0 AND volume > 1.5x 20-period average.
# Short when: Alligator jaws > teeth > lips AND Bear Power < 0 AND volume > 1.5x 20-period average.
# Exit when Alligator lines re-cross (jaws crosses teeth) or volume drops below average.
# Uses Williams Alligator for trend identification, Elder Ray for bull/bear power, volume for confirmation.
# Target: 50-150 total trades over 4 years (12-37/year) with controlled frequency.

name = "6h_WilliamsAlligator_1dElderRay_Volume"
timeframe = "6h"
leverage = 1.0

def smma(data, period):
    """Smoothed Moving Average (SMMA) - same as used in Williams Alligator"""
    if len(data) < period:
        return np.full_like(data, np.nan, dtype=float)
    result = np.full_like(data, np.nan, dtype=float)
    # First value is simple moving average
    result[period-1] = np.mean(data[:period])
    # Subsequent values: (prev * (period-1) + current) / period
    for i in range(period, len(data)):
        result[i] = (result[i-1] * (period-1) + data[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Daily data for Elder Ray and volume
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 20:
        return np.zeros(n)
    
    # 60-period EMA for 1d (Elder Ray uses EMA13 typically, but we'll use EMA60 for smoother 1d trend)
    close_d = df_d['close'].values
    ema60_d = pd.Series(close_d).ewm(span=60, adjust=False, min_periods=60).mean().values
    
    # Elder Ray: Bull Power = High - EMA60, Bear Power = Low - EMA60
    bull_power = df_d['high'].values - ema60_d
    bear_power = df_d['low'].values - ema60_d
    
    # Volume filter: current volume > 1.5x 20-period average
    volume_d = df_d['volume'].values
    vol_ma20_d = pd.Series(volume_d).rolling(window=20, min_periods=20).mean().values
    volume_filter_d = volume_d > (1.5 * vol_ma20_d)
    volume_filter = align_htf_to_ltf(prices, df_d, volume_filter_d)
    
    # Align Elder Ray to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_d, bear_power)
    
    # Williams Alligator on 6h timeframe
    # Jaws: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    jaws = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Alligator alignment: jaws < teeth < lips = bullish alignment
    # jaws > teeth > lips = bearish alignment
    alligator_bullish = (jaws < teeth) & (teeth < lips)
    alligator_bearish = (jaws > teeth) & (teeth > lips)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(60, 20)  # Sufficient warmup for EMA60 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Alligator bullish alignment AND Bull Power > 0 AND volume filter
            long_cond = alligator_bullish[i] and (bull_power_aligned[i] > 0) and volume_filter[i]
            # Short conditions: Alligator bearish alignment AND Bear Power < 0 AND volume filter
            short_cond = alligator_bearish[i] and (bear_power_aligned[i] < 0) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator loses bullish alignment OR Bull Power <= 0
            if not (alligator_bullish[i] and (bull_power_aligned[i] > 0)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator loses bearish alignment OR Bear Power >= 0
            if not (alligator_bearish[i] and (bear_power_aligned[i] < 0)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals