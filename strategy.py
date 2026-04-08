#!/usr/bin/env python3
"""
6H Camarilla Pivot + Daily EMA + Volume Confirmation v2
Hypothesis: Camarilla pivot levels from daily timeframe provide high-probability reversal zones.
Fade at R3/S3 levels with trend alignment (daily EMA) and volume confirmation.
Works in both bull/bear markets by fading extremes in ranging markets and following trends.
Target: 15-35 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_daily_ema_volume_v2"
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
    
    # Daily data for Camarilla pivots and EMA
    df_1d = get_htf_data(prices, '1d')
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1)
    prev_high = df_1d['high'].shift(1)
    prev_low = df_1d['low'].shift(1)
    
    # Calculate Camarilla levels
    range_val = prev_high - prev_low
    camarilla_h5 = prev_close + (range_val * 1.1 / 2)
    camarilla_h4 = prev_close + (range_val * 1.1 / 4)
    camarilla_h3 = prev_close + (range_val * 1.1 / 6)
    camarilla_l3 = prev_close - (range_val * 1.1 / 6)
    camarilla_l4 = prev_close - (range_val * 1.1 / 4)
    camarilla_l5 = prev_close - (range_val * 1.1 / 2)
    
    # Daily EMA(21) for trend filter
    ema_21 = df_1d['close'].ewm(span=21, adjust=False, min_periods=21).mean()
    
    # Align to 6h timeframe
    h3_6h = align_htf_to_ltf(prices, df_1d, camarilla_h3.values)
    l3_6h = align_htf_to_ltf(prices, df_1d, camarilla_l3.values)
    ema_21_6h = align_htf_to_ltf(prices, df_1d, ema_21.values)
    
    # Volume filter (>1.3x 20-period average on 6h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(h3_6h[i]) or np.isnan(l3_6h[i]) or 
            np.isnan(ema_21_6h[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches H3 or trend reverses
            if close[i] >= h3_6h[i] or close[i] < ema_21_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches L3 or trend reverses
            if close[i] <= l3_6h[i] or close[i] > ema_21_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Fade long at L3 with trend alignment (price above EMA = bullish bias)
            if (close[i] <= l3_6h[i] and 
                close[i] > ema_21_6h[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Fade short at H3 with trend alignment (price below EMA = bearish bias)
            elif (close[i] >= h3_6h[i] and 
                  close[i] < ema_21_6h[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals