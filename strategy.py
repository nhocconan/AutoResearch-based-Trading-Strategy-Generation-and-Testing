#!/usr/bin/env python3
"""
12h_camarilla_pivot_1w_trend_volume_v1
Hypothesis: On 12h timeframe, use Camarilla pivot levels from weekly timeframe for entry signals, filtered by weekly EMA trend and volume confirmation.
- In uptrend (price > weekly EMA50): long at S3 (support), exit at S4 (breakdown) or trend reversal
- In downtrend (price < weekly EMA50): short at R3 (resistance), exit at R4 (breakout) or trend reversal
Volume confirms genuine tests of pivot levels. Weekly timeframe provides stronger trend filter suitable for 12h chart.
Target: 12-37 trades/year (50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1w_trend_volume_v1"
timeframe = "12h"
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
    
    # Weekly data for pivot calculation and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    ema_50 = df_1w['close'].ewm(span=50, adjust=False).mean()
    
    # Align weekly EMA50 to 12h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50.values)
    
    # Calculate Camarilla pivot levels from previous week
    # Using previous week's OHLC to avoid look-ahead
    prev_close = df_1w['close'].shift(1)
    prev_high = df_1w['high'].shift(1)
    prev_low = df_1w['low'].shift(1)
    
    # Pivot point
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    
    # Camarilla levels
    # Resistance levels
    r1 = pivot + (range_val * 1.1 / 12)
    r2 = pivot + (range_val * 1.1 / 6)
    r3 = pivot + (range_val * 1.1 / 4)
    r4 = pivot + (range_val * 1.1 / 2)
    # Support levels
    s1 = pivot - (range_val * 1.1 / 12)
    s2 = pivot - (range_val * 1.1 / 6)
    s3 = pivot - (range_val * 1.1 / 4)
    s4 = pivot - (range_val * 1.1 / 2)
    
    # Align all levels to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot.values)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1.values)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2.values)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3.values)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4.values)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1.values)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2.values)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3.values)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4.values)
    
    # Volume confirmation (2-period average on 12h = 1 day)
    vol_ma = pd.Series(volume).rolling(window=2, min_periods=2).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(ema_50_aligned[i]) or np.isnan(pivot_aligned[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0 or
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below S4 (breakdown) or trend turns bearish
            if close[i] < s4_aligned[i] or close[i] < ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price breaks above R4 (breakout) or trend turns bullish
            if close[i] > r4_aligned[i] or close[i] > ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price tests S3 with volume in uptrend
            if (abs(close[i] - s3_aligned[i]) / s3_aligned[i] < 0.005 and  # within 0.5% of S3
                vol_confirm and 
                close[i] > ema_50_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price tests R3 with volume in downtrend
            elif (abs(close[i] - r3_aligned[i]) / r3_aligned[i] < 0.005 and  # within 0.5% of R3
                  vol_confirm and 
                  close[i] < ema_50_aligned[i]):
                position = -1
                signals[i] = -0.25
            # Long breakout: price breaks above R4 with volume in uptrend
            elif (close[i] > r4_aligned[i] and
                  vol_confirm and 
                  close[i] > ema_50_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short breakdown: price breaks below S4 with volume in downtrend
            elif (close[i] < s4_aligned[i] and
                  vol_confirm and 
                  close[i] < ema_50_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals