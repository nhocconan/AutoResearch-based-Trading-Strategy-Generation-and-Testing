#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d trend filter and volume confirmation
# Long when Alligator jaws < teeth < lips (bullish alignment) + 1d EMA50 up + volume > 1.5x average
# Short when Alligator jaws > teeth > lips (bearish alignment) + 1d EMA50 down + volume > 1.5x average
# Williams Alligator identifies trend phases; 1d EMA50 filters for higher timeframe trend; volume confirms conviction
# Targets 50-150 total trades over 4 years (12-37/year) to balance opportunity and cost

name = "6h_WilliamsAlligator_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data once for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA(50) for trend filter
    daily_close = df_1d['close'].values
    ema50_1d = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Williams Alligator: SMMA (Smoothed Moving Average) with offsets
    # Jaws: SMMA(13, 8), Teeth: SMMA(8, 5), Lips: SMMA(5, 3)
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: (prev * (period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaws = smma(close, 13)  # Blue line
    teeth = smma(close, 8)  # Red line
    lips = smma(close, 5)   # Green line
    
    # Average volume for confirmation
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams Alligator alignment
        bullish_alignment = jaws[i] < teeth[i] and teeth[i] < lips[i]
        bearish_alignment = jaws[i] > teeth[i] and teeth[i] > lips[i]
        
        # Daily trend filter
        daily_uptrend = ema50_1d_aligned[i] > ema50_1d_aligned[i-1]
        daily_downtrend = ema50_1d_aligned[i] < ema50_1d_aligned[i-1]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Enter long: bullish alignment + daily uptrend + volume confirmation
            if bullish_alignment and daily_uptrend and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish alignment + daily downtrend + volume confirmation
            elif bearish_alignment and daily_downtrend and vol_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: alignment breaks or daily trend reverses
            if not bullish_alignment or not daily_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: alignment breaks or daily trend reverses
            if not bearish_alignment or not daily_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals