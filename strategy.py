#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 12h Trend + Volume Spike
# Long when price > Alligator teeth, 12h EMA50 rising, volume > 1.5x avg
# Short when price < Alligator teeth, 12h EMA50 falling, volume > 1.5x avg
# Alligator identifies trend, EMA50 confirms higher timeframe direction, volume filters weak moves
# Targets 50-150 total trades over 4 years (12-37/year) to balance accuracy and frequency

name = "6h_WilliamsAlligator_12hTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data once for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Williams Alligator (13,8,5) with SMMA (Smoothed Moving Average)
    def smma(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (prev * (period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(high, 13)  # Blue line (13-period)
    teeth = smma(low, 8)   # Red line (8-period)
    lips = smma(close, 5)  # Green line (5-period)
    
    # Volume spike filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for Alligator and EMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator condition: price alignment
        price_above_teeth = close[i] > teeth[i]
        price_below_teeth = close[i] < teeth[i]
        
        if position == 0:
            # Enter long: price > teeth, 12h EMA50 rising, volume spike
            if (price_above_teeth and 
                ema50_12h_aligned[i] > ema50_12h_aligned[i-1] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price < teeth, 12h EMA50 falling, volume spike
            elif (price_below_teeth and 
                  ema50_12h_aligned[i] < ema50_12h_aligned[i-1] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price < teeth or 12h EMA50 falling or no volume spike
            if (price_below_teeth or 
                ema50_12h_aligned[i] < ema50_12h_aligned[i-1] or 
                not vol_spike[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price > teeth or 12h EMA50 rising or no volume spike
            if (price_above_teeth or 
                ema50_12h_aligned[i] > ema50_12h_aligned[i-1] or 
                not vol_spike[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals