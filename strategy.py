#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data once (HTF)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points using previous week's data
    # Convert daily to weekly by sampling every 5th day (approximate)
    # For simplicity, we'll use daily pivots but with longer lookback for stability
    # Calculate pivot points using previous 5-day range for weekly context
    high_5d = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
    low_5d = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
    close_5d = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values
    
    # Shift to avoid look-ahead: use previous 5-day data
    prev_high_5d = np.roll(high_5d, 1)
    prev_low_5d = np.roll(low_5d, 1)
    prev_close_5d = np.roll(close_5d, 1)
    prev_high_5d[0] = np.nan
    prev_low_5d[0] = np.nan
    prev_close_5d[0] = np.nan
    
    # Weekly-style pivot points (using 5-day aggregation)
    pp_5d = (prev_high_5d + prev_low_5d + prev_close_5d) / 3
    r1_5d = 2 * pp_5d - prev_low_5d
    s1_5d = 2 * pp_5d - prev_high_5d
    r2_5d = pp_5d + (prev_high_5d - prev_low_5d)
    s2_5d = pp_5d - (prev_high_5d - prev_low_5d)
    r3_5d = pp_5d + 2 * (prev_high_5d - prev_low_5d)
    s3_5d = pp_5d - 2 * (prev_high_5d - prev_low_5d)
    
    # 5-day EMA for trend filter (more stable than daily)
    ema5_5d = pd.Series(close_5d).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Align 5-day pivots and EMA to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp_5d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_5d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_5d)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2_5d)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2_5d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_5d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_5d)
    ema5_aligned = align_htf_to_ltf(prices, df_1d, ema5_5d)
    
    # Volume spike filter (24-period average on 6h data ≈ 6 days)
    volume = prices['volume'].values
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any data is not ready
        if (np.isnan(pp_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(r2_aligned[i]) or 
            np.isnan(s2_aligned[i]) or 
            np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(ema5_aligned[i]) or 
            np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_24[i]
        pp = pp_aligned[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        r2 = r2_aligned[i]
        s2 = s2_aligned[i]
        r3 = r3_aligned[i]
        s3 = s3_aligned[i]
        ema5 = ema5_aligned[i]
        
        if position == 0:
            # Long: price breaks above S3 (deep value) with volume spike and above EMA5
            if price < s3 and vol > 2.0 * vol_ma and price > ema5:
                signals[i] = 0.25
                position = 1
            # Short: price breaks above R3 (overbought) with volume spike and below EMA5
            elif price > r3 and vol > 2.0 * vol_ma and price < ema5:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price returns to central pivot or opposite extreme
            if position == 1:  # Long position
                if price > pp or price < s3:  # Return to mean or oversold
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:  # Short position
                if price < pp or price > r3:  # Return to mean or overbought
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Pivot_S3_R3_Reversal_5dEMA5_Volume_Spike"
timeframe = "6h"
leverage = 1.0