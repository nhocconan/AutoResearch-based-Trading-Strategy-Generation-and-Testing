#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_Pivot_R3S3_Breakout_Volume_ATRFilter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # === Weekly Pivot Points (previous week) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Previous week's values for pivot calculation
    prev_high = np.roll(high_1w, 1)
    prev_low = np.roll(low_1w, 1)
    prev_close = np.roll(close_1w, 1)
    
    # Set first values to avoid look-ahead
    prev_high[0] = high_1w[0]
    prev_low[0] = low_1w[0]
    prev_close[0] = close_1w[0]
    
    # Classic pivot (same for all pivot types)
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # Weekly R3 and S3 levels
    r3 = pivot + (range_val * 1.1)
    s3 = pivot - (range_val * 1.1)
    
    # Align to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # === Volume Confirmation ===
    volume = prices['volume'].values
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # === ATR Stop Loss (6h) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr_series = pd.Series(tr)
    atr = atr_series.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Get values
        close_val = prices['close'].iloc[i]
        vol_ratio_val = vol_ratio[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        atr_val = atr[i]
        
        # Skip if any value is NaN
        if (np.isnan(vol_ratio_val) or np.isnan(r3_val) or np.isnan(s3_val) or 
            np.isnan(atr_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Breakout at R3/S3 with volume confirmation
            if close_val > r3_val and vol_ratio_val > 2.0:
                # Break above R3
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif close_val < s3_val and vol_ratio_val > 2.0:
                # Break below S3
                signals[i] = -0.25
                position = -1
                entry_price = close_val
        
        elif position == 1:
            # Long exit: stop loss or return to S3
            if close_val <= entry_price - 2.0 * atr_val:
                # Stop loss hit
                signals[i] = 0.0
                position = 0
            elif close_val < s3_val:
                # Return to S3
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: stop loss or return to R3
            if close_val >= entry_price + 2.0 * atr_val:
                # Stop loss hit
                signals[i] = 0.0
                position = 0
            elif close_val > r3_val:
                # Return to R3
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals