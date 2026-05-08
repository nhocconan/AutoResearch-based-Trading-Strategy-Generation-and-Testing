#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Williams Alligator with daily trend filter and volume spike
# Long when price > Alligator Jaw (13-period SMMA), daily EMA(34) uptrend, volume spike
# Short when price < Alligator Jaw, daily EMA(34) downtrend, volume spike
# Alligator uses smoothed moving averages (SMMA) for teeth (8), jaw (13), lips (5)
# Daily EMA provides trend filter; volume confirms breakout strength
# Targets 50-150 total trades over 4 years (12-37/year) to balance opportunity and cost

name = "12h_WilliamsAlligator_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

def smma(arr, period):
    """Smoothed Moving Average (SMMA)"""
    if len(arr) < period:
        return np.full_like(arr, np.nan, dtype=float)
    result = np.full_like(arr, np.nan, dtype=float)
    # First value is simple moving average
    result[period-1] = np.mean(arr[:period])
    # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CLOSE) / period
    for i in range(period, len(arr)):
        result[i] = (result[i-1] * (period-1) + arr[i]) / period
    return result

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
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA(34) for trend filter
    daily_close = df_1d['close'].values
    ema34_1d = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Williams Alligator components (SMMA)
    jaw = smma(close, 13)   # Jaw (13-period SMMA)
    teeth = smma(close, 8)  # Teeth (8-period SMMA)
    lips = smma(close, 5)   # Lips (5-period SMMA)
    
    # Align daily EMA to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), jaw)
    # Note: For alignment, we use close prices as reference since Alligator is price-based
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        jaw_val = jaw_aligned[i]
        ema34_1d_val = ema34_1d_aligned[i]
        price = close[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price > Jaw, daily uptrend, volume spike
            if price > jaw_val and price > ema34_1d_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: price < Jaw, daily downtrend, volume spike
            elif price < jaw_val and price < ema34_1d_val and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price falls below Jaw or daily trend turns down
            if price < jaw_val or price < ema34_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises above Jaw or daily trend turns up
            if price > jaw_val or price > ema34_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals