#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_Pivot_R3S3_Breakout_Volume_TrendFilter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 3:  # Need at least 3 weeks for pivot calculation
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need at least 30 days for EMA
        return np.zeros(n)
    
    # === Weekly: Calculate Pivot Points (R3, S3) ===
    # Use previous week's OHLC for current week's pivots
    prev_week_high = np.roll(df_1w['high'].values, 1)
    prev_week_low = np.roll(df_1w['low'].values, 1)
    prev_week_close = np.roll(df_1w['close'].values, 1)
    
    # Set first week's values to NaN
    prev_week_high[0] = np.nan
    prev_week_low[0] = np.nan
    prev_week_close[0] = np.nan
    
    # Calculate pivot point and support/resistance levels
    pivot_point = (prev_week_high + prev_week_low + prev_week_close) / 3
    r1 = 2 * pivot_point - prev_week_low
    s1 = 2 * pivot_point - prev_week_high
    r2 = pivot_point + (prev_week_high - prev_week_low)
    s2 = pivot_point - (prev_week_high - prev_week_low)
    r3 = prev_week_high + 2 * (pivot_point - prev_week_low)
    s3 = prev_week_low - 2 * (prev_week_high - pivot_point)
    
    # Align weekly pivots to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # === Daily: Trend filter (EMA 50) ===
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # === 6h: Volume ratio (current vs 20-period average) ===
    close = prices['close'].values
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_processes=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup
        # Get values
        close_val = close[i]
        ema_val = ema_50_aligned[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema_val) or np.isnan(r3_val) or 
            np.isnan(s3_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above daily EMA50 (uptrend), breaks above weekly R3, volume confirmation
            if (close_val > ema_val and      # Uptrend filter
                close_val > r3_val and       # Break above weekly R3
                vol_ratio_val > 1.5):        # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: Price below daily EMA50 (downtrend), breaks below weekly S3, volume confirmation
            elif (close_val < ema_val and    # Downtrend filter
                  close_val < s3_val and     # Break below weekly S3
                  vol_ratio_val > 1.5):      # Volume confirmation
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price drops below daily EMA50 or breaks below weekly S3 (reversal)
            if close_val < ema_val or close_val < s3_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price rises above daily EMA50 or breaks above weekly R3 (reversal)
            if close_val > ema_val or close_val > r3_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals