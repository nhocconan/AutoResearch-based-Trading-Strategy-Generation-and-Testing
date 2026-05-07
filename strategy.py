#!/usr/bin/env python3
name = "6h_Pivot_R3S3_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h and 1d data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_12h) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d Pivot levels (Standard Pivot): R3, S3 from previous day
    # Standard Pivot: P = (H + L + C) / 3
    # R3 = H + 2*(P - L) = 3*H - 2*L
    # S3 = L - 2*(H - P) = 3*L - 2*H
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    prev_close_1d = df_1d['close'].shift(1).values
    pivot_p_1d = (prev_high_1d + prev_low_1d + prev_close_1d) / 3
    pivot_r3_1d = 3 * prev_high_1d - 2 * prev_low_1d
    pivot_s3_1d = 3 * prev_low_1d - 2 * prev_high_1d
    
    # Align 1d Pivot levels to 6h timeframe
    pivot_r3_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_r3_1d)
    pivot_s3_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_s3_1d)
    
    # 12h ADX for trend filter (ADX > 25 indicates strong trend)
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smooth using Wilder's smoothing (alpha = 1/period)
        atr = np.zeros_like(tr)
        plus_dm_smooth = np.zeros_like(plus_dm)
        minus_dm_smooth = np.zeros_like(minus_dm)
        
        atr[period] = np.nansum(tr[1:period+1])
        plus_dm_smooth[period] = np.nansum(plus_dm[1:period+1])
        minus_dm_smooth[period] = np.nansum(plus_dm[1:period+1])
        
        for i in range(period+1, len(tr)):
            atr[i] = atr[i-1] - (atr[i-1] / period) + tr[i]
            plus_dm_smooth[i] = plus_dm_smooth[i-1] - (plus_dm_smooth[i-1] / period) + plus_dm[i]
            minus_dm_smooth[i] = minus_dm_smooth[i-1] - (minus_dm_smooth[i-1] / period) + minus_dm[i]
        
        # Avoid division by zero
        plus_di = np.where(atr != 0, 100 * plus_dm_smooth / atr, 0)
        minus_di = np.where(atr != 0, 100 * minus_dm_smooth / atr, 0)
        dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
        
        # Smooth DX to get ADX
        adx = np.zeros_like(dx)
        adx[2*period] = np.nansum(dx[period+1:2*period+1]) / period
        for i in range(2*period+1, len(dx)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    adx_12h = calculate_adx(high_12h, low_12h, close_12h, 14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # 6h volume spike: > 2.0x 20-period average (adjusted for 6h)
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike_6h = volume > 2.0 * vol_ma_6h
    
    # 6h EMA20 for entry filter
    ema20_6h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 34)  # Wait for ADX and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_r3_1d_aligned[i]) or np.isnan(pivot_s3_1d_aligned[i]) or 
            np.isnan(adx_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above R3 with volume spike, strong trend (ADX > 25), and price above EMA20
            if (close[i] > pivot_r3_1d_aligned[i] and vol_spike_6h[i] and 
                adx_12h_aligned[i] > 25 and close[i] > ema20_6h[i]):
                signals[i] = 0.25
                position = 1
            # Short: Break below S3 with volume spike, strong trend (ADX > 25), and price below EMA20
            elif (close[i] < pivot_s3_1d_aligned[i] and vol_spike_6h[i] and 
                  adx_12h_aligned[i] > 25 and close[i] < ema20_6h[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price below S3 or trend weakening (ADX < 20)
            if close[i] < pivot_s3_1d_aligned[i] or adx_12h_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price above R3 or trend weakening (ADX < 20)
            if close[i] > pivot_r3_1d_aligned[i] or adx_12h_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Standard Pivot R3/S3 levels act as strong S/R on 1d timeframe.
# Breakouts with volume confirmation and 12h trend alignment capture strong moves.
# Works in bull/bear: long breakouts in uptrend, short breakdowns in downtrend.
# Volume spike filters false breakouts. ADX filter ensures trending conditions.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.