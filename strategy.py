#!/usr/bin/env python3
"""
6h_ADX_WilliamsAlligator_v1
Hypothesis: Combine ADX(14) trend strength with Williams Alligator (SMAs 13,8,5) on 6h timeframe. 
Enter long when ADX > 25 (strong trend) + price > Alligator Jaw (13-period SMA) + Alligator aligned bullish (Teeth > Lips). 
Enter short when ADX > 25 + price < Alligator Jaw + Alligator aligned bearish (Teeth < Lips). 
Exit when ADX < 20 (weakening trend) or Alligator alignment reverses. 
Position size: 0.25. Uses 1-day EMA50 as higher timeframe filter to avoid counter-trend trades. 
Target: 50-150 total trades over 4 years = 12-37/year. 
Works in bull (ADX up + price > Jaw) and bear (ADX up + price < Jaw) markets when trend is strong.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for HTF trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator on 6h: Jaw (13), Teeth (8), Lips (5) SMAs
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    
    # ADX calculation
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Smoothed values with Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[1:period+1])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period = 14
    tr_sum = wilder_smooth(tr, period)
    plus_dm_sum = wilder_smooth(plus_dm, period)
    minus_dm_sum = wilder_smooth(minus_dm, period)
    
    # Avoid division by zero
    divisor = np.where(tr_sum != 0, tr_sum, 1e-10)
    plus_di = 100 * plus_dm_sum / divisor
    minus_di = 100 * minus_dm_sum / divisor
    
    dx = np.full_like(plus_di, np.nan)
    dx_divisor = np.where((plus_di + minus_di) != 0, (plus_di + minus_di), 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / dx_divisor
    
    adx = wilder_smooth(dx, period)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Alligator (13) and ADX (14*2)
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d HTF trend (bullish = price above daily EMA50)
        htf_1d_bullish = close[i] > ema_50_1d_aligned[i]
        htf_1d_bearish = close[i] < ema_50_1d_aligned[i]
        
        # Alligator alignment
        alligator_bullish = teeth[i] > lips[i]  # Teeth above Lips
        alligator_bearish = teeth[i] < lips[i]  # Teeth below Lips
        
        # ADX trend strength
        strong_trend = adx[i] > 25
        weak_trend = adx[i] < 20
        
        if position == 0:
            # Long setup: strong trend + price > Jaw + Alligator bullish + 1d uptrend
            long_setup = (strong_trend and 
                         close[i] > jaw[i] and 
                         alligator_bullish and 
                         htf_1d_bullish)
            
            # Short setup: strong trend + price < Jaw + Alligator bearish + 1d downtrend
            short_setup = (strong_trend and 
                          close[i] < jaw[i] and 
                          alligator_bearish and 
                          htf_1d_bearish)
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: weakening trend OR Alligator alignment turns bearish OR 1d trend turns bearish
            if (weak_trend or not alligator_bullish or not htf_1d_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: weakening trend OR Alligator alignment turns bullish OR 1d trend turns bullish
            if (weak_trend or alligator_bullish or htf_1d_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ADX_WilliamsAlligator_v1"
timeframe = "6h"
leverage = 1.0