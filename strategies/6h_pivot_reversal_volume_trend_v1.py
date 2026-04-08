#!/usr/bin/env python3
"""
6h Pivot Reversal with Volume Surge and 1d Trend Filter
Hypothesis: Daily pivot points provide institutional support/resistance.
On 6b timeframe, look for price rejection at pivot levels (R1/S1, R2/S2) with
volume surge indicating institutional interest, filtered by 1d EMA trend.
Works in bull/bear by using adaptive pivot levels and trend alignment.
Target: 15-25 trades/year on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_pivot_reversal_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for pivots and trend
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Standard pivot points (not Camarilla)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    # R2 = P + (H - L)
    # S2 = P - (H - L)
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    r2_1d = pivot_1d + (high_1d - low_1d)
    s2_1d = pivot_1d - (high_1d - low_1d)
    
    # Align pivot levels to 6h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: current volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_surge = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or np.isnan(r2_1d_aligned[i]) or
            np.isnan(s2_1d_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_surge[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below S1 OR trend turns bearish
            if (close[i] <= s1_1d_aligned[i] or 
                close[i] <= ema_50_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above R1 OR trend turns bullish
            if (close[i] >= r1_1d_aligned[i] or 
                close[i] >= ema_50_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price rejects S1 with volume surge and uptrend
            # Rejection = price touches/goes below S1 then closes back above it
            if (low[i] <= s1_1d_aligned[i] and 
                close[i] > s1_1d_aligned[i] and
                close[i] > ema_50_1d_aligned[i] and
                vol_surge[i]):
                position = 1
                signals[i] = 0.25
            # Short: price rejects R1 with volume surge and downtrend
            elif (high[i] >= r1_1d_aligned[i] and 
                  close[i] < r1_1d_aligned[i] and
                  close[i] < ema_50_1d_aligned[i] and
                  vol_surge[i]):
                position = -1
                signals[i] = -0.25
            # Long: price breaks above R2 with volume surge and uptrend (momentum)
            elif (close[i] >= r2_1d_aligned[i] and 
                  close[i] > ema_50_1d_aligned[i] and
                  vol_surge[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below S2 with volume surge and downtrend (momentum)
            elif (close[i] <= s2_1d_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and
                  vol_surge[i]):
                position = -1
                signals[i] = -0.25
    
    return signals