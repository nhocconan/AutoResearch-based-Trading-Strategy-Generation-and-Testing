#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Pivot_R3S3_Fade_With_Volume_Confirmation"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop for pivot levels and volume average
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # 1d high, low, close for pivot calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot points and support/resistance levels
    pivot = (high_1d + low_1d + close_1d) / 3
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    r2 = pivot + (high_1d - low_1d)
    s2 = pivot - (high_1d - low_1d)
    r3 = high_1d + 2 * (pivot - low_1d)
    s3 = low_1d - 2 * (high_1d - pivot)
    r4 = 3 * pivot - 2 * low_1d
    s4 = 3 * pivot - 2 * high_1d
    
    # Align pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # 1d average volume (20-period) for volume confirmation
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # 6h ATR for exit (14-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Pre-compute hour filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get aligned values
        pivot_val = pivot_aligned[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        r4_val = r4_aligned[i]
        s4_val = s4_aligned[i]
        vol_avg = vol_avg_1d_aligned[i]
        current_atr = atr[i]
        current_close = prices['close'].iloc[i]
        current_volume = prices['volume'].iloc[i]
        
        # Skip if any value is NaN
        if np.isnan(pivot_val) or np.isnan(r3_val) or np.isnan(s3_val) or \
           np.isnan(r4_val) or np.isnan(s4_val) or np.isnan(vol_avg) or np.isnan(current_atr):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.5x daily average volume
        vol_spike = current_volume > 1.5 * vol_avg
        
        if position == 0:
            # Fade at R3/S3: price touches R3/S3 with volume spike
            if current_close <= r3_val and current_close >= s3_val and vol_spike:
                # Check if price is closer to R3 or S3 for direction
                dist_to_r3 = abs(current_close - r3_val)
                dist_to_s3 = abs(current_close - s3_val)
                if dist_to_r3 < dist_to_s3:
                    # Near R3, look for rejection (price < pivot)
                    if current_close < pivot_val:
                        signals[i] = -0.25  # short
                        position = -1
                        entry_price = current_close
                else:
                    # Near S3, look for bounce (price > pivot)
                    if current_close > pivot_val:
                        signals[i] = 0.25   # long
                        position = 1
                        entry_price = current_close
            # Breakout at R4/S4: price breaks R4/S4 with volume spike
            elif current_close >= r4_val and vol_spike:
                signals[i] = 0.25   # long breakout
                position = 1
                entry_price = current_close
            elif current_close <= s4_val and vol_spike:
                signals[i] = -0.25  # short breakdown
                position = -1
                entry_price = current_close
        
        elif position == 1:
            # Long exit: price below pivot or ATR stop loss
            if current_close < pivot_val:
                signals[i] = 0.0
                position = 0
            elif current_close < entry_price - 2.5 * current_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above pivot or ATR stop loss
            if current_close > pivot_val:
                signals[i] = 0.0
                position = 0
            elif current_close > entry_price + 2.5 * current_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals