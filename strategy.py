#!/usr/bin/env python3
"""
4h_12h_camarilla_breakout_volume_v1
Strategy: 4h Camarilla Pivot breakout with volume confirmation and 12h trend filter
Timeframe: 4h
Leverage: 1.0
Hypothesis: Uses 4h price breakout above/below Camarilla Pivot levels (calculated from previous 12h range) confirmed by volume spike (>1.5x average volume) and filtered by 12h EMA25 trend direction. Camarilla levels work well in ranging and trending markets, providing clear support/resistance levels. Volume confirmation filters out weak breakouts. Trend filter ensures trades align with higher timeframe momentum. Designed to work in both bull and bear markets by capturing breakouts in the direction of the 12h trend. Target: 80-180 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_camarilla_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla Pivot levels for 4h using previous 12h bar
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate pivot and ranges
    pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    range_12h = high_12h - low_12h
    
    # Camarilla levels
    # Resistance levels
    r1_12h = close_12h + (range_12h * 1.1 / 12)
    r2_12h = close_12h + (range_12h * 1.1 / 6)
    r3_12h = close_12h + (range_12h * 1.1 / 4)
    r4_12h = close_12h + (range_12h * 1.1 / 2)
    # Support levels
    s1_12h = close_12h - (range_12h * 1.1 / 12)
    s2_12h = close_12h - (range_12h * 1.1 / 6)
    s3_12h = close_12h - (range_12h * 1.1 / 4)
    s4_12h = close_12h - (range_12h * 1.1 / 2)
    
    # Align Camarilla levels to 4h timeframe (using previous 12h bar's levels)
    r1_4h = align_htf_to_ltf(prices, df_12h, r1_12h)
    r2_4h = align_htf_to_ltf(prices, df_12h, r2_12h)
    r3_4h = align_htf_to_ltf(prices, df_12h, r3_12h)
    r4_4h = align_htf_to_ltf(prices, df_12h, r4_12h)
    s1_4h = align_htf_to_ltf(prices, df_12h, s1_12h)
    s2_4h = align_htf_to_ltf(prices, df_12h, s2_12h)
    s3_4h = align_htf_to_ltf(prices, df_12h, s3_12h)
    s4_4h = align_htf_to_ltf(prices, df_12h, s4_12h)
    
    # 12h EMA25 for trend filter
    ema_25_12h = pd.Series(close_12h).ewm(span=25, adjust=False, min_periods=25).mean().values
    ema_25_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_25_12h)
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(25, n):  # Start after EMA warmup
        # Skip if any required data is invalid
        if (np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or 
            np.isnan(ema_25_12h_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Trend filter: price above/below 12h EMA25
        uptrend_12h = price_close > ema_25_12h_aligned[i]
        downtrend_12h = price_close < ema_25_12h_aligned[i]
        
        # Breakout conditions (using Camarilla levels from previous 12h bar)
        breakout_r3 = price_close > r3_4h[i]  # Break above R3
        breakdown_s3 = price_close < s3_4h[i]  # Break below S3
        
        # Volume confirmation
        vol_confirmed = vol_spike[i]
        
        # Long: break above R3 with volume in uptrend
        long_signal = breakout_r3 and vol_confirmed and uptrend_12h
        
        # Short: break below S3 with volume in downtrend
        short_signal = breakdown_s3 and vol_confirmed and downtrend_12h
        
        # Exit when price returns to opposite S1/R1 level (mean reversion to mean)
        exit_long = position == 1 and price_close < s1_4h[i]
        exit_short = position == -1 and price_close > r1_4h[i]
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals