#!/usr/bin/env python3
"""
1h_Pivot_R2_S2_Breakout_1dTrend_VolumeFilter
Hypothesis: Daily pivot levels R2/S2 act as strong support/resistance.
Breakouts above R2 or below S2 with volume confirmation and daily EMA trend
filter capture institutional moves. Works in both bull and bear by following
institutional flow. Uses 4h for directional bias and 1h only for entry timing.
Target: 15-37 trades/year (60-150 total over 4 years) to balance opportunity and fee drag.
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
    
    # 1-day data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla pivot
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla levels: R2 = C + (H-L)*1.1/6, S2 = C - (H-L)*1.1/6
    r2 = prev_close + (prev_high - prev_low) * 1.1 / 6
    s2 = prev_close - (prev_high - prev_low) * 1.1 / 6
    
    # Align to 1h timeframe (waits for daily bar to close)
    r2_1h = align_htf_to_ltf(prices, df_1d, r2)
    s2_1h = align_htf_to_ltf(prices, df_1d, s2)
    
    # Volume filter: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    # 1-day EMA trend filter
    ema_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_1h = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 4h directional bias filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    ema_4h = pd.Series(df_4h['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_4h_1h = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    signals = np.zeros(n)
    position = 0
    bars_since_entry = 0
    
    start_idx = 20  # Warmup for volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(r2_1h[i]) or np.isnan(s2_1h[i]) or
            np.isnan(volume_filter[i]) or np.isnan(ema_1d_1h[i]) or
            np.isnan(ema_4h_1h[i])):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        price = close[i]
        r2_val = r2_1h[i]
        s2_val = s2_1h[i]
        vol_ok = volume_filter[i]
        ema_trend = ema_1d_1h[i]
        ema_bias = ema_4h_1h[i]
        
        if position == 0:
            # Long: break above R2 with volume in uptrend, aligned with 4h bias
            if price > r2_val and vol_ok and price > ema_trend and price > ema_bias:
                signals[i] = 0.20
                position = 1
                bars_since_entry = 0
            # Short: break below S2 with volume in downtrend, aligned with 4h bias
            elif price < s2_val and vol_ok and price < ema_trend and price < ema_bias:
                signals[i] = -0.20
                position = -1
                bars_since_entry = 0
        
        elif position == 1:
            bars_since_entry += 1
            # Minimum holding period: 6 bars (6 hours)
            if bars_since_entry < 6:
                signals[i] = 0.20
            else:
                signals[i] = 0.20
                # Exit: price returns to S2 or trend reverses
                if price < s2_val or price < ema_trend:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
        
        elif position == -1:
            bars_since_entry += 1
            # Minimum holding period: 6 bars (6 hours)
            if bars_since_entry < 6:
                signals[i] = -0.20
            else:
                signals[i] = -0.20
                # Exit: price returns to R2 or trend reverses
                if price > r2_val or price > ema_trend:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
    
    return signals

name = "1h_Pivot_R2_S2_Breakout_1dTrend_VolumeFilter"
timeframe = "1h"
leverage = 1.0