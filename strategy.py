#!/usr/bin/env python3
"""
6h Camarilla Pivot with 1d EMA Filter and Volume Confirmation
Hypothesis: Camarilla pivot levels from daily timeframe identify key support/resistance.
Fade at R3/S3 (mean reversion in range) and breakout continuation at R4/S4 (trend continuation).
Combined with 1d EMA trend filter to align with higher timeframe direction and volume to confirm.
Works in bull/bear by using pivot structure that adapts to volatility and EMA filter for trend alignment.
Target: 20-40 trades/year on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_1d_ema_volume_v1"
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
    
    # 1d OHLC for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    # Camarilla formulas:
    # Range = High - Low
    # R4 = Close + Range * 1.1/2
    # R3 = Close + Range * 1.1/4
    # R2 = Close + Range * 1.1/6
    # R1 = Close + Range * 1.1/12
    # S1 = Close - Range * 1.1/12
    # S2 = Close - Range * 1.1/6
    # S3 = Close - Range * 1.1/4
    # S4 = Close - Range * 1.1/2
    range_1d = high_1d - low_1d
    r4_1d = close_1d + range_1d * 1.1 / 2
    r3_1d = close_1d + range_1d * 1.1 / 4
    s3_1d = close_1d - range_1d * 1.1 / 4
    s4_1d = close_1d - range_1d * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe (shifted by 1 for completed daily bars only)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter (>1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below S3 OR trend turns bearish
            if (close[i] <= s3_1d_aligned[i] or 
                close[i] <= ema_50_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above R3 OR trend turns bullish
            if (close[i] >= r3_1d_aligned[i] or 
                close[i] >= ema_50_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price breaks above R4 with uptrend and volume (breakout continuation)
            if (close[i] >= r4_1d_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below S4 with downtrend and volume (breakout continuation)
            elif (close[i] <= s4_1d_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
            # Long: price bounces off S3 with uptrend and volume (mean reversion fade)
            elif (close[i] <= s3_1d_aligned[i] and 
                  close[i] > ema_50_1d_aligned[i] and 
                  vol_filter[i] and
                  i > 0 and close[i] > close[i-1]):  # confirming bounce
                position = 1
                signals[i] = 0.25
            # Short: price bounces off R3 with downtrend and volume (mean reversion fade)
            elif (close[i] >= r3_1d_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  vol_filter[i] and
                  i > 0 and close[i] < close[i-1]):  # confirming bounce
                position = -1
                signals[i] = -0.25
    
    return signals