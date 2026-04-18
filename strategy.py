#!/usr/bin/env python3
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
    
    # Get 1d data for weekly pivot and daily ATR (HTF)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate weekly pivot levels (using previous week's high, low, close)
    # We'll calculate weekly data by resampling daily data (but using actual weekly bars via resample is not allowed)
    # Instead, we calculate pivots on a rolling weekly basis using the last 5 days
    # This is an approximation but avoids look-ahead by using only past data
    def calculate_weekly_pivot(high_arr, low_arr, close_arr):
        # For each point, use the previous 5 days (if available) to calculate weekly pivot
        # This ensures no look-ahead
        weekly_high = np.full_like(high_arr, np.nan)
        weekly_low = np.full_like(low_arr, np.nan)
        weekly_close = np.full_like(close_arr, np.nan)
        
        for i in range(len(high_arr)):
            if i >= 5:
                weekly_high[i] = np.max(high_arr[i-5:i])
                weekly_low[i] = np.min(low_arr[i-5:i])
                weekly_close[i] = close_arr[i-1]  # Previous day's close as weekly close
        
        # Calculate pivot points: P = (H + L + C)/3
        weekly_p = (weekly_high + weekly_low + weekly_close) / 3.0
        # R1 = 2*P - L, S1 = 2*P - H
        weekly_r1 = 2 * weekly_p - weekly_low
        weekly_s1 = 2 * weekly_p - weekly_high
        # R2 = P + (H - L), S2 = P - (H - L)
        weekly_r2 = weekly_p + (weekly_high - weekly_low)
        weekly_s2 = weekly_p - (weekly_high - weekly_low)
        # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
        weekly_r3 = weekly_high + 2 * (weekly_p - weekly_low)
        weekly_s3 = weekly_low - 2 * (weekly_high - weekly_p)
        
        return weekly_p, weekly_r1, weekly_s1, weekly_r2, weekly_s2, weekly_r3, weekly_s3
    
    # Calculate weekly pivot levels on daily data
    wp, wr1, ws1, wr2, ws2, wr3, ws3 = calculate_weekly_pivot(high_1d, low_1d, close_1d)
    
    # Align weekly pivot levels to 6h (wait for the weekly bar to complete - using 5-day lookback means we need to align properly)
    # Since we used previous 5 days, we align without additional delay as the weekly pivot is based on completed week
    wp_aligned = align_htf_to_ltf(prices, df_1d, wp)
    wr1_aligned = align_htf_to_ltf(prices, df_1d, wr1)
    ws1_aligned = align_htf_to_ltf(prices, df_1d, ws1)
    wr2_aligned = align_htf_to_ltf(prices, df_1d, wr2)
    ws2_aligned = align_htf_to_ltf(prices, df_1d, ws2)
    wr3_aligned = align_htf_to_ltf(prices, df_1d, wr3)
    ws3_aligned = align_htf_to_ltf(prices, df_1d, ws3)
    
    # Calculate 1d ATR (14-period) for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align ATR to 6h
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 1d volume spike (volume > 1.5x 20-period average)
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (1.5 * vol_ma_1d)
    volume_spike_6h = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # wait for enough data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(wr3_aligned[i]) or
            np.isnan(ws3_aligned[i]) or
            np.isnan(atr_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when ATR is above its 50-period average (avoid low volatility chop)
        if i >= 50:
            atr_ma_1d = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
            atr_ma_6h = align_htf_to_ltf(prices, df_1d, atr_ma_1d)
            vol_filter = atr_1d_aligned[i] > atr_ma_6h[i] if not np.isnan(atr_ma_6h[i]) else False
        else:
            vol_filter = False
        
        if position == 0:
            # Long: price touches or goes below S3 with volume and volatility filter (mean reversion from extreme)
            if close[i] <= ws3_aligned[i] and volume_spike_6h[i] and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price touches or goes above R3 with volume and volatility filter (mean reversion from extreme)
            elif close[i] >= wr3_aligned[i] and volume_spike_6h[i] and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price reaches midpoint between S3 and S2 or shows weakness
            midpoint_s2_s3 = (ws2_aligned[i] + ws3_aligned[i]) / 2.0
            if close[i] >= midpoint_s2_s3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches midpoint between R3 and R2 or shows weakness
            midpoint_r2_r3 = (wr2_aligned[i] + wr3_aligned[i]) / 2.0
            if close[i] <= midpoint_r2_r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_R3_S3_MeanReversion_Volume"
timeframe = "6h"
leverage = 1.0