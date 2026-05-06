#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using Camarilla R3/S3 pivot levels from 1d with volume confirmation and choppiness regime filter
# Long when price crosses above Camarilla R3 level AND volume > 1.3 * avg_volume(20) AND choppiness index < 50 (trending market)
# Short when price crosses below Camarilla S3 level AND volume > 1.3 * avg_volume(20) AND choppiness index < 50 (trending market)
# Exit when price returns to Camarilla Pivot level (PP) or opposite extreme (S3/R3)
# Uses discrete sizing 0.25 to balance profit potential and drawdown control
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Camarilla levels from 1d provide strong intraday support/resistance that works in both bull and bear markets
# Volume confirmation filters out low-conviction breakouts
# Choppiness regime filter ensures we only trade in trending conditions (avoids choppy whipsaws)
# Works in bull markets (breakout continuations) and bear markets (breakdown continuations)

name = "4h_1dCamarilla_R3S3_Breakout_Volume_ChopFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:  # Need at least 1 completed daily bar
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for 1d
    # Camarilla formulas: PP = (H+L+C)/3, R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    pp_1d = typical_price_1d  # Simplified: using typical price as pivot
    r3_1d = close_1d + (range_1d * 1.1 / 2.0)
    s3_1d = close_1d - (range_1d * 1.1 / 2.0)
    
    # Align 1d Camarilla levels to 4h timeframe (wait for completed 1d bar)
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Calculate volume confirmation: volume > 1.3 * 20-period average volume on 4h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * avg_volume_20)
    
    # Calculate choppiness index regime filter on 4h (trending when CHOP < 50)
    # CHOP = 100 * log10(sum(ATR(14)) / log10(highest_high - lowest_low)) / log10(14)
    tr14 = np.maximum(np.abs(high[1:] - low[:-1]), np.absolute(np.abs(close[1:] - close[:-1])))
    tr14 = np.concatenate([[np.nan], tr14])  # align with original length
    atr14 = pd.Series(tr14).rolling(window=14, min_periods=14).mean().values
    
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_14 = highest_high_14 - lowest_low_14
    range_14 = np.where(range_14 == 0, 1e-10, range_14)
    
    chop = 100 * np.log10(atr14 * 14 / np.log10(range_14)) / np.log10(14)
    chop[np.isnan(chop)] = 50  # default to neutral when not enough data
    chop_filter = chop < 50  # trending market
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(pp_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or np.isnan(avg_volume_20[i]) or
            np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price crosses above R3 level, volume spike, trending market
            if (close[i] > r3_1d_aligned[i] and close[i-1] <= r3_1d_aligned[i-1] and 
                volume_confirm[i] and chop_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price crosses below S3 level, volume spike, trending market
            elif (close[i] < s3_1d_aligned[i] and close[i-1] >= s3_1d_aligned[i-1] and 
                  volume_confirm[i] and chop_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to PP level or below S3 (opposite extreme)
            if close[i] <= pp_1d_aligned[i] or close[i] <= s3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to PP level or above R3 (opposite extreme)
            if close[i] >= pp_1d_aligned[i] or close[i] >= r3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals