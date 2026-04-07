#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Weekly Pivot Fade with Volume Divergence
# Hypothesis: Weekly pivot levels (R3/S3, R4/S4) act as strong support/resistance.
# Price often reverses from R3/S3 (fade) but breaks through R4/S4 (continuation).
# Uses volume divergence: weakening volume on approach to R3/S3 increases reversal probability.
# Works in bull/bear by fading extremes in range and following breakouts in trend.
# Target: 20-50 trades/year (80-200 over 4 years).

name = "6h_weekly_pivot_fade_volume_divergence_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # Calculate weekly data (previous week's OHLC)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Shift by 1 to use previous week's data (avoid look-ahead)
    prev_weekly_high = np.roll(weekly_high, 1)
    prev_weekly_low = np.roll(weekly_low, 1)
    prev_weekly_close = np.roll(weekly_close, 1)
    prev_weekly_high[0] = prev_weekly_high[1] if len(prev_weekly_high) > 1 else 0
    prev_weekly_low[0] = prev_weekly_low[1] if len(prev_weekly_low) > 1 else 0
    prev_weekly_close[0] = prev_weekly_close[1] if len(prev_weekly_close) > 1 else 0
    
    # Calculate weekly pivot points (R3/S3, R4/S4)
    weekly_range = prev_weekly_high - prev_weekly_low
    weekly_pivot = (prev_weekly_high + prev_weekly_low + prev_weekly_close) / 3.0
    weekly_r3 = weekly_pivot + (weekly_range * 1.1)
    weekly_s3 = weekly_pivot - (weekly_range * 1.1)
    weekly_r4 = weekly_pivot + (weekly_range * 1.5)
    weekly_s4 = weekly_pivot - (weekly_range * 1.5)
    
    # Align to 6h timeframe (use previous week's levels)
    weekly_r3_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r3)
    weekly_s3_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s3)
    weekly_r4_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r4)
    weekly_s4_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s4)
    
    # Volume divergence: decreasing volume on approach to extremes
    vol_series = pd.Series(volume)
    vol_ma_10 = vol_series.rolling(window=10, min_periods=10).mean().values
    vol_ma_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    # Volume weakening when short MA < long MA
    vol_weakening = vol_ma_10 < vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(weekly_r3_aligned[i]) or np.isnan(weekly_s3_aligned[i]) or
            np.isnan(weekly_r4_aligned[i]) or np.isnan(weekly_s4_aligned[i]) or
            np.isnan(vol_ma_10[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches R4 (take profit) or shows weakness at R3 with volume divergence
            if (close[i] >= weekly_r4_aligned[i] or 
                (close[i] >= weekly_r3_aligned[i] and vol_weakening[i])):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price reaches S4 (take profit) or shows weakness at S3 with volume divergence
            if (close[i] <= weekly_s4_aligned[i] or 
                (close[i] <= weekly_s3_aligned[i] and vol_weakening[i])):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price rejects S3 with volume weakening (bullish divergence)
            if ((low[i] <= weekly_s3_aligned[i] or close[i] <= weekly_s3_aligned[i]) and 
                vol_weakening[i]):
                position = 1
                signals[i] = 0.25
            # Short: price rejects R3 with volume weakening (bearish divergence)
            elif ((high[i] >= weekly_r3_aligned[i] or close[i] >= weekly_r3_aligned[i]) and 
                  vol_weakening[i]):
                position = -1
                signals[i] = -0.25
    
    return signals