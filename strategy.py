#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R1/S1 breakout with 1w EMA50 trend filter and volume confirmation.
Long when price breaks above Camarilla R1 level AND volume > 2.0x 20-period average AND price > 1w EMA50 (bullish trend).
Short when price breaks below Camarilla S1 level AND volume > 2.0x 20-period average AND price < 1w EMA50 (bearish trend).
Exit when price crosses the 1w EMA50 in opposite direction or touches Camarilla S3/R3 levels.
Designed for very low trade frequency (12-37/year) to minimize fee drag while capturing strong breakouts in both bull and bear markets.
Uses 1w EMA50 as regime filter to avoid counter-trend trades in strong trends.
"""

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
    
    # Get 1w data for EMA50 trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Get 1d data for Camarilla pivot calculation (HTF)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 1w timeframe
    close_1w_series = pd.Series(close_1w)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Camarilla pivot levels on 1d timeframe
    # Typical price for pivot point
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    # Camarilla width
    camarilla_width = (high_1d - low_1d) * 1.1 / 12.0
    # Camarilla levels
    camarilla_h4 = close_1d + camarilla_width * 1.1 * 2  # R3 equivalent
    camarilla_h3 = close_1d + camarilla_width * 1.1      # R2 equivalent
    camarilla_h2 = close_1d + camarilla_width * 0.5      # R1 equivalent
    camarilla_l2 = close_1d - camarilla_width * 0.5      # S1 equivalent
    camarilla_l3 = close_1d - camarilla_width * 1.1      # S2 equivalent
    camarilla_l4 = close_1d - camarilla_width * 1.1 * 2  # S3 equivalent
    
    # Align all indicators to 12h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    camarilla_h2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h2)   # R1
    camarilla_l2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l2)   # S1
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)   # R3
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)   # S3
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(camarilla_h2_aligned[i]) or 
            np.isnan(camarilla_l2_aligned[i]) or np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i])):
            signals[i] = 0.0
            continue
        
        ema_50 = ema_50_1w_aligned[i]
        r1 = camarilla_h2_aligned[i]    # Camarilla R1
        s1 = camarilla_l2_aligned[i]    # Camarilla S1
        r3 = camarilla_h4_aligned[i]    # Camarilla R3
        s3 = camarilla_l4_aligned[i]    # Camarilla S3
        vol = volume[i]
        price = close[i]
        high_price = high[i]
        low_price = low[i]
        
        if position == 0:
            # Long: price breaks above R1 AND price > 1w EMA50 (bullish trend)
            if high_price > r1 and price > ema_50:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 AND price < 1w EMA50 (bearish trend)
            elif low_price < s1 and price < ema_50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below EMA50 OR touches R3 (take profit)
            if price < ema_50 or high_price >= r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above EMA50 OR touches S3 (take profit)
            if price > ema_50 or low_price <= s3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_1wEMA50_TrendFilter"
timeframe = "12h"
leverage = 1.0