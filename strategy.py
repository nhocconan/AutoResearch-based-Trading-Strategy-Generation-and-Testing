#!/usr/bin/env python3
# 1h_Camarilla_Pivot_Breakout_4hTrend_1dVolume
# Hypothesis: Use Camarilla pivot levels on 1h for precise entry/exit, with 4h trend filter and 1d volume confirmation.
# Enter long when price breaks above Camarilla R3 level with volume, short when breaks below S3 level with volume, only in direction of 4h trend.
# Exit when price returns to Camarilla pivot (P) or trend reverses.
# Designed for low frequency (15-30 trades/year) by using 1h for entry timing, 4h for trend, 1d for volume filter.

name = "1h_Camarilla_Pivot_Breakout_4hTrend_1dVolume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 1h data for Camarilla pivot levels ===
    # Calculate pivot points from previous bar (standard method)
    # We'll use the previous bar's high, low, close to calculate today's pivot
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    
    # Camarilla levels
    r3 = pivot + (range_val * 1.1 / 2.0)  # R3 = P + 1.1*(H-L)/2
    r2 = pivot + (range_val * 1.1 / 4.0)  # R2 = P + 1.1*(H-L)/4
    r1 = pivot + (range_val * 1.1 / 6.0)  # R1 = P + 1.1*(H-L)/6
    s1 = pivot - (range_val * 1.1 / 6.0)  # S1 = P - 1.1*(H-L)/6
    s2 = pivot - (range_val * 1.1 / 4.0)  # S2 = P - 1.1*(H-L)/4
    s3 = pivot - (range_val * 1.1 / 2.0)  # S3 = P - 1.1*(H-L)/2
    
    # === 4h data for trend filter ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    # EMA(50) on 4h for trend
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # === 1d data for volume confirmation ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    # Volume MA(20) on 1d
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # 1h volume MA(20) for entry confirmation
    vol_ma_20_1h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(pivot[i]) or np.isnan(r3[i]) or np.isnan(s3[i]) or
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(vol_ma_20_1h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 4h EMA50
        trend_up = close[i] > ema_50_4h_aligned[i]
        trend_down = close[i] < ema_50_4h_aligned[i]
        
        # Volume confirmation: both 1h and 1d volume above average
        vol_1h_ok = volume[i] > vol_ma_20_1h[i]
        vol_1d_ok = vol_ma_20_1d_aligned[i] > 0 and volume[i] > vol_ma_20_1d_aligned[i] * 0.5  # Allow some flexibility
        
        if position == 0:
            # LONG: Price breaks above R3 with volume, in uptrend
            if close[i] > r3[i] and vol_1h_ok and vol_1d_ok and trend_up:
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below S3 with volume, in downtrend
            elif close[i] < s3[i] and vol_1h_ok and vol_1d_ok and trend_down:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # EXIT LONG: Price returns to pivot or trend reverses
            if close[i] <= pivot[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price returns to pivot or trend reverses
            if close[i] >= pivot[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals