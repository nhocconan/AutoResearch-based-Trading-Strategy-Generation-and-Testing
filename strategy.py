#!/usr/bin/env python3
"""
6h Camarilla Pivot + 1d EMA Trend + Volume Filter
Hypothesis: Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) provide
high-probability reversal/continuation signals. Filtered by 1d EMA trend and volume spike.
Works in bull/bear by using volatility-adjusted positions and trend alignment. Targets 15-35 trades/year on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_1d_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivot and EMA
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate previous day's Camarilla levels
    # Using (H+L+C)/3 as pivot (standard for Camarilla)
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels: R4 = C + (H-L)*1.5, R3 = C + (H-L)*1.25, etc.
    r4_1d = close_1d + range_1d * 1.5
    r3_1d = close_1d + range_1d * 1.25
    s3_1d = close_1d - range_1d * 1.25
    s4_1d = close_1d - range_1d * 1.5
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators to 6h timeframe (shifted by 1 bar for completed bars only)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 6h ATR(20) for volatility filter
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Volume filter (>1.8x 30-period average)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_filter = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or np.isnan(r4_1d_aligned[i]) or 
            np.isnan(s4_1d_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below S3 OR trend reverses
            if (close[i] <= s3_1d_aligned[i] or 
                close[i] < ema_50_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above R3 OR trend reverses
            if (close[i] >= r3_1d_aligned[i] or 
                close[i] > ema_50_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Mean reversion long at S3 with trend alignment and volume
            if (close[i] <= s3_1d_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Mean reversion short at R3 with trend alignment and volume
            elif (close[i] >= r3_1d_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
            # Breakout long at R4 with trend alignment and volume
            elif (close[i] >= r4_1d_aligned[i] and 
                  close[i] > ema_50_1d_aligned[i] and 
                  vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Breakdown short at S4 with trend alignment and volume
            elif (close[i] <= s4_1d_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals