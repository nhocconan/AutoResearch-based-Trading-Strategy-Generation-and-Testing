#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Pivot_R1S1_MomentumBreakout_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 150:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === Daily Pivot Points (previous day) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values for pivot calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # Pivot point
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # Key levels: R1 and S1
    r1 = pivot + (range_val * 1.1 / 12)
    s1 = pivot - (range_val * 1.1 / 12)
    
    # Align to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # === 6h Momentum and Volume ===
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 30-period EMA for trend filter (2.5 days)
    close_series = pd.Series(close)
    ema30 = close_series.ewm(span=30, adjust=False, min_periods=30).mean().values
    
    # Volume ratio (10-period average) - shorter for sensitivity
    vol_series = pd.Series(volume)
    vol_ma10 = vol_series.rolling(window=10, min_periods=10).mean().values
    vol_ratio = volume / np.where(vol_ma10 > 0, vol_ma10, np.nan)
    
    # Momentum: 3-period ROC
    roc3 = np.zeros_like(close)
    roc3[3:] = (close[3:] - close[:-3]) / close[:-3] * 100
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):
        # Get values
        close_val = close[i]
        roc_val = roc3[i]
        vol_ratio_val = vol_ratio[i]
        ema30_val = ema30[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        pivot_val = pivot_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(roc_val) or np.isnan(vol_ratio_val) or np.isnan(ema30_val) or 
            np.isnan(r1_val) or np.isnan(s1_val) or np.isnan(pivot_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above R1 with positive momentum and volume
            if (close_val > r1_val and 
                roc_val > 0.3 and 
                vol_ratio_val > 1.2 and
                close_val > ema30_val):
                signals[i] = 0.25
                position = 1
            # Short: Break below S1 with negative momentum and volume
            elif (close_val < s1_val and 
                  roc_val < -0.3 and 
                  vol_ratio_val > 1.2 and
                  close_val < ema30_val):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns below pivot or momentum turns negative
            if close_val < pivot_val or roc_val < -0.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns above pivot or momentum turns positive
            if close_val > pivot_val or roc_val > 0.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals