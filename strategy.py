#!/usr/bin/env python3
"""
12h_Pivot_R1S1_R2S2_Breakout_Volume_Trend
Hypothesis: Daily Pivot levels (R1,S1,R2,S2) from 1-day timeframe act as strong support/resistance.
Breakouts above R2 or below S2 with volume confirmation and daily EMA trend filter capture
institutional move initiation. Works in bull/bear by following institutional flow.
Target: 12-37 trades/year (48-148 total over 4 years) to balance opportunity and fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day data for Pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Pivot
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Pivot point and support/resistance levels
    pivot = (prev_high + prev_low + prev_close) / 3
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    r2 = pivot + (prev_high - prev_low)
    s2 = pivot - (prev_high - prev_low)
    
    # Align to 12h timeframe (waits for 1-day bar to close)
    r2_12h = align_htf_to_ltf(prices, df_1d, r2)
    s2_12h = align_htf_to_ltf(prices, df_1d, s2)
    r1_12h = align_htf_to_ltf(prices, df_1d, r1)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume filter: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    # 1-day EMA trend filter
    ema_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_12h = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0
    bars_since_entry = 0
    
    start_idx = 20  # Warmup for volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(r2_12h[i]) or np.isnan(s2_12h[i]) or
            np.isnan(volume_filter[i]) or np.isnan(ema_1d_12h[i])):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        price = close[i]
        r2_val = r2_12h[i]
        s2_val = s2_12h[i]
        r1_val = r1_12h[i]
        s1_val = s1_12h[i]
        vol_ok = volume_filter[i]
        ema_trend = ema_1d_12h[i]
        
        if position == 0:
            # Long: break above R2 with volume in uptrend
            if price > r2_val and vol_ok and price > ema_trend:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: break below S2 with volume in downtrend
            elif price < s2_val and vol_ok and price < ema_trend:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        
        elif position == 1:
            bars_since_entry += 1
            # Minimum holding period: 2 bars (1 day)
            if bars_since_entry < 2:
                signals[i] = 0.25
            else:
                signals[i] = 0.25
                # Exit: price returns to S1 or trend reverses
                if price < s1_val or price < ema_trend:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
        
        elif position == -1:
            bars_since_entry += 1
            # Minimum holding period: 2 bars (1 day)
            if bars_since_entry < 2:
                signals[i] = -0.25
            else:
                signals[i] = -0.25
                # Exit: price returns to R1 or trend reverses
                if price > r1_val or price > ema_trend:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
    
    return signals

name = "12h_Pivot_R1S1_R2S2_Breakout_Volume_Trend"
timeframe = "12h"
leverage = 1.0