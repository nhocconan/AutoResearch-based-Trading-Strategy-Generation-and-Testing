#!/usr/bin/env python3
"""
6h Camarilla Pivot + 1d EMA Trend + Volume Confirmation
Hypothesis: Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) 
provide high-probability reversal/breakout points. Trend filtered by daily EMA(21) 
ensures directional alignment with higher timeframe. Volume > 1.5x average confirms 
institutional participation. Designed for 6h timeframe to balance trade frequency 
and signal quality in both bull and bear markets.
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
    
    # 1d data for Camarilla pivots and EMA
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day
    # Typical price = (H + L + C) / 3
    typical = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    range_ = df_1d['high'] - df_1d['low']
    
    # Camarilla levels: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, etc.
    r4 = df_1d['close'] + range_ * 1.1 / 2
    r3 = df_1d['close'] + range_ * 1.1 / 4
    s3 = df_1d['close'] - range_ * 1.1 / 4
    s4 = df_1d['close'] - range_ * 1.1 / 2
    
    # Align to 6h timeframe (shifted by 1 for completed bars only)
    r4_6h = align_ftf_to_ltf(prices, df_1d, r4.values)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3.values)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3.values)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4.values)
    
    # 1d EMA(21) for trend filter
    ema_21 = df_1d['close'].ewm(span=21, adjust=False).mean().values
    ema_21_6h = align_htf_to_ltf(prices, df_1d, ema_21)
    
    # Volume filter (>1.5x 24-period average on 6h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(24, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or 
            np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]) or
            np.isnan(ema_21_6h[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below S3 (mean reversion failure) or trend reverses
            if close[i] < s3_6h[i] or close[i] < ema_21_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above R3 (mean reversion failure) or trend reverses
            if close[i] > r3_6h[i] or close[i] > ema_21_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Mean reversion long at S3 with trend alignment
            if (close[i] <= s3_6h[i] and 
                close[i] > s4_6h[i] and  # Above S4 to avoid breakdown
                close[i] > ema_21_6h[i] and  # Uptrend filter
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Mean reversion short at R3 with trend alignment
            elif (close[i] >= r3_6h[i] and 
                  close[i] < r4_6h[i] and  # Below R4 to avoid breakout
                  close[i] < ema_21_6h[i] and  # Downtrend filter
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
            # Breakout long at R4 with trend alignment
            elif (close[i] >= r4_6h[i] and 
                  close[i] > ema_21_6h[i] and 
                  vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Breakout short at S4 with trend alignment
            elif (close[i] <= s4_6h[i] and 
                  close[i] < ema_21_6h[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals

def align_ftf_to_ltf(prices, df_htf, values):
    """Helper function for HTF alignment (using align_htf_to_ltf internally)"""
    return align_htf_to_ltf(prices, df_htf, values)