#!/usr/bin/env python3
"""
1d_camarilla_pivot_1w_trend_volume_v1
Hypothesis: On 1d timeframe, use Camarilla pivot levels from 1w for entry signals, filtered by 1w EMA trend and volume confirmation.
- In uptrend (price > 1w EMA50): long at S3 (support) with volume, exit at S4 breakdown or trend reversal
- In downtrend (price < 1w EMA50): short at R3 (resistance) with volume, exit at R4 breakout or trend reversal
Volume confirms genuine tests of pivot levels. This strategy fades at S3/R3 in ranging markets
and captures breakouts at S4/R4 in trending markets, adapting to both bull and bear regimes.
Target: 7-25 trades/year (~30-100 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_camarilla_pivot_1w_trend_volume_v1"
timeframe = "1d"
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
    
    # 1w data for pivot calculation and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 for trend filter
    ema_50 = df_1w['close'].ewm(span=50, adjust=False).mean()
    
    # Align 1w EMA50 to 1d timeframe
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
    
    # Align all levels to 1d timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot.values)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1.values)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2.values)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3.values)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4.values)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1.values)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2.values)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3.values)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4.values)
    
    # Volume confirmation (5-period average on 1d = 5 days)
    vol_ma = pd.Series(volume).rolling(window=5, min_periods=5).mean().values
    
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
            if (close[i] <= s3_aligned[i] * 1.005 and close[i] >= s3_aligned[i] * 0.995 and  # near S3
                vol_confirm and 
                close[i] > ema_50_aligned[i]):  # uptrend filter
                position = 1
                signals[i] = 0.25
            # Short entry: price tests R3 with volume in downtrend
            elif (close[i] >= r3_aligned[i] * 0.995 and close[i] <= r3_aligned[i] * 1.005 and  # near R3
                  vol_confirm and 
                  close[i] < ema_50_aligned[i]):  # downtrend filter
                position = -1
                signals[i] = -0.25
            # Long breakout: price breaks above R4 with volume in uptrend
            elif (close[i] > r4_aligned[i] and
                  vol_confirm and 
                  close[i] > ema_50_aligned[i]):  # uptrend filter
                position = 1
                signals[i] = 0.25
            # Short breakdown: price breaks below S4 with volume in downtrend
            elif (close[i] < s4_aligned[i] and
                  vol_confirm and 
                  close[i] < ema_50_aligned[i]):  # downtrend filter
                position = -1
                signals[i] = -0.25
    
    return signals