#!/usr/bin/env python3
"""
6h_camarilla_pivot_1d_ema_volume_v2
Hypothesis: On 6h timeframe, enter long when price bounces from S3/S4 Camarilla levels (1d) with EMA(50) support and volume confirmation; enter short when price rejects R3/R4 with EMA(50) resistance. Uses 12h trend filter (EMA50 slope) to avoid counter-trend trades. Designed for 15-30 trades/year to minimize fee fade while capturing mean reversion at institutional pivot levels in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_1d_ema_volume_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # EMA(50) for dynamic support/resistance
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Pivot point
    pivot = (prev_high + prev_low + prev_close) / 3.0
    
    # Camarilla levels
    s1 = pivot - (prev_high - prev_low) * 1.0 / 12
    s2 = pivot - (prev_high - prev_low) * 2.0 / 12
    s3 = pivot - (prev_high - prev_low) * 3.0 / 12
    s4 = pivot - (prev_high - prev_low) * 4.0 / 12
    r1 = pivot + (prev_high - prev_low) * 1.0 / 12
    r2 = pivot + (prev_high - prev_low) * 2.0 / 12
    r3 = pivot + (prev_high - prev_low) * 3.0 / 12
    r4 = pivot + (prev_high - prev_low) * 4.0 / 12
    
    # Align Camarilla levels to 6h timeframe
    pivot_6h = align_htf_to_ltf(prices, df_1d, pivot)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    
    # 12h trend filter: EMA50 slope
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Slope of EMA50 (12h): positive = uptrend, negative = downtrend
    ema_slope = np.diff(ema_50_12h_aligned, prepend=ema_50_12h_aligned[0])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if data not available
        if (np.isnan(close[i]) or np.isnan(ema_50[i]) or np.isnan(vol_ma[i]) or
            np.isnan(pivot_6h[i]) or np.isnan(s3_6h[i]) or np.isnan(s4_6h[i]) or
            np.isnan(r3_6h[i]) or np.isnan(r4_6h[i]) or np.isnan(ema_slope[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: above average volume
        vol_ok = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price closes below EMA50 or touches S4
            if close[i] < ema_50[i] or close[i] <= s4_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above EMA50 or touches R4
            if close[i] > ema_50[i] or close[i] >= r4_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Long: price touches/bounces from S3 or S4 with EMA50 support and uptrend
                if ((close[i] <= s3_6h[i] * 1.001 and close[i] >= s3_6h[i] * 0.999) or
                    (close[i] <= s4_6h[i] * 1.001 and close[i] >= s4_6h[i] * 0.999)) and \
                   close[i] > ema_50[i] and ema_slope[i] > 0:
                    position = 1
                    signals[i] = 0.25
                # Short: price touches/rejects from R3 or R4 with EMA50 resistance and downtrend
                elif ((close[i] >= r3_6h[i] * 0.999 and close[i] <= r3_6h[i] * 1.001) or
                      (close[i] >= r4_6h[i] * 0.999 and close[i] <= r4_6h[i] * 1.001)) and \
                     close[i] < ema_50[i] and ema_slope[i] < 0:
                    position = -1
                    signals[i] = -0.25
    
    return signals