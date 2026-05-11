# 6h_1d_Market_Profile_Value_Area
# Hypothesis: 6h chart with daily market profile value area (POC/VA) and 1d EMA trend filter.
# In ranging markets, price reverts to value area POC; in trending markets, breaks above/below VA with trend continuation.
# Works in both bull/bear by adapting to market structure via value area and trend filter.
# Target: 50-150 total trades over 4 years (12-37/year).

#!/usr/bin/env python3
name = "6h_1d_Market_Profile_Value_Area"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for market profile and trend
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Value Area (VA) and Point of Control (POC) for each day
    # Using volume-weighted price distribution (simplified: VWAP over day)
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap = (typical_price * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    
    # VA boundaries: 1 standard deviation around VWAP
    price_deviation = typical_price - vwap
    # Variance of price deviation weighted by volume
    var = ((price_deviation ** 2) * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    std_dev = np.sqrt(var)
    
    vwap_val = vwap.values
    std_val = std_dev.values
    
    va_high = vwap_val + std_val
    va_low = vwap_val - std_val
    poc = vwap_val  # POC at VWAP
    
    # Trend filter: EMA50 > EMA200 for uptrend
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    trend_up = ema50_1d > ema200_1d
    trend_down = ema50_1d < ema200_1d
    
    # Align all to 6h
    poc_aligned = align_htf_to_ltf(prices, df_1d, poc)
    va_high_aligned = align_htf_to_ltf(prices, df_1d, va_high)
    va_low_aligned = align_htf_to_ltf(prices, df_1d, va_low)
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, trend_up)
    trend_down_aligned = align_htf_to_ltf(prices, df_1d, trend_down)
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma20 = np.zeros(n)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i < 20:
            vol_ma20[i] = vol_sum / (i+1) if i > 0 else 0
        else:
            vol_ma20[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 50)
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(poc_aligned[i]) or np.isnan(va_high_aligned[i]) or np.isnan(va_low_aligned[i]) or
            np.isnan(trend_up_aligned[i]) or np.isnan(trend_down_aligned[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price below VA low in uptrend with volume surge (mean reversion to value)
            if (close[i] < va_low_aligned[i] and 
                trend_up_aligned[i] and 
                volume[i] > 1.3 * vol_ma20[i]):
                signals[i] = 0.25
                position = 1
            # Short: price above VA high in downtrend with volume surge (mean reversion to value)
            elif (close[i] > va_high_aligned[i] and 
                  trend_down_aligned[i] and 
                  volume[i] > 1.3 * vol_ma20[i]):
                signals[i] = -0.25
                position = -1
            # Long breakout: price above VA high in uptrend with volume surge
            elif (close[i] > va_high_aligned[i] and 
                  trend_up_aligned[i] and 
                  volume[i] > 1.3 * vol_ma20[i]):
                signals[i] = 0.25
                position = 1
            # Short breakout: price below VA low in downtrend with volume surge
            elif (close[i] < va_low_aligned[i] and 
                  trend_down_aligned[i] and 
                  volume[i] > 1.3 * vol_ma20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reaches POC or trend changes
            if (close[i] >= poc_aligned[i] or not trend_up_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reaches POC or trend changes
            if (close[i] <= poc_aligned[i] or not trend_down_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals