#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d trend filter
# Uses Alligator (Jaw/Teeth/Lips) to detect trends and avoid whipsaws
# 1d EMA200 filter ensures we only trade in the direction of the higher timeframe trend
# Works in bull/bear by aligning with dominant trend and using Alligator for entry/exit
# Target: 50-150 total trades over 4 years (12-37/year)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams Alligator (13,8,5 smoothed with 8,5,3)
    # Jaw (blue): 13-period SMMA smoothed by 8
    # Teeth (red): 8-period SMMA smoothed by 5
    # Lips (green): 5-period SMMA smoothed by 3
    
    def smma(arr, period):
        """Smoothed Moving Average"""
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full_like(arr, np.nan, dtype=float)
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (Prev SMMA * (period-1) + Current Price) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    # Calculate Alligator lines
    jaw = smma(smma(close, 13), 8)   # 13-period smoothed by 8
    teeth = smma(smma(close, 8), 5)  # 8-period smoothed by 5
    lips = smma(smma(close, 5), 3)   # 5-period smoothed by 3
    
    # 1d EMA200 for trend filter (higher timeframe trend)
    df_1d = get_htf_data(prices, '1d')
    ema_200_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for Alligator calculation
    start = 50  # sufficient for SMMA smoothing
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_200_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Trend filter: only trade when price is above/below 1d EMA200
        if price > ema_200_1d_aligned[i]:
            # Bullish bias: look for long signals
            if position == 0:
                # Enter long when Lips > Teeth > Jaw (bullish alignment)
                if lips[i] > teeth[i] and teeth[i] > jaw[i]:
                    position = 1
                    signals[i] = position_size
                else:
                    signals[i] = 0.0
            elif position == 1:
                # Stay long
                signals[i] = position_size
            elif position == -1:
                # Exit short (reverse to flat)
                position = 0
                signals[i] = 0.0
        else:
            # Bearish bias: look for short signals
            if position == 0:
                # Enter short when Lips < Teeth < Jaw (bearish alignment)
                if lips[i] < teeth[i] and teeth[i] < jaw[i]:
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
            elif position == -1:
                # Stay short
                signals[i] = -position_size
            elif position == 1:
                # Exit long (reverse to flat)
                position = 0
                signals[i] = 0.0
    
    return signals

name = "6h_Williams_Alligator_1dEMA200_Filter"
timeframe = "6h"
leverage = 1.0