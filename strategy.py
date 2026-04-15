#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d EMA50 Trend Filter
# Uses Williams Alligator (Smoothed Moving Average with 5,8,13 periods) on 6h timeframe
# to identify trend direction and entry points. Filters trades with 1d EMA50 to ensure
# alignment with higher timeframe trend. Works in both bull and bear markets by
# following the higher timeframe trend. Target: 50-150 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams Alligator components on 6h (Smoothed Moving Average)
    # Jaw (13-period SMMA), Teeth (8-period SMMA), Lips (5-period SMMA)
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is simple moving average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CLOSE) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Load 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 1d
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            continue
        
        # Bullish alignment: Lips > Teeth > Jaw (alligator mouth opening up)
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        # Bearish alignment: Lips < Teeth < Jaw (alligator mouth opening down)
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Long entry: Bullish alignment + price above 1d EMA50
        if bullish_alignment and close[i] > ema_50_1d_aligned[i] and position <= 0:
            position = 1
            signals[i] = base_size
        
        # Short entry: Bearish alignment + price below 1d EMA50
        elif bearish_alignment and close[i] < ema_50_1d_aligned[i] and position >= 0:
            position = -1
            signals[i] = -base_size
        
        # Exit: Opposite alignment or price crosses 1d EMA50 in opposite direction
        elif position == 1 and (not bullish_alignment or close[i] < ema_50_1d_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (not bearish_alignment or close[i] > ema_50_1d_aligned[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_WilliamsAlligator_1dEMA50"
timeframe = "6h"
leverage = 1.0