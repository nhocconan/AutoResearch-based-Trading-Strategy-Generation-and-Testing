#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 1d Williams %R filter and volume confirmation
- Williams %R(14) on 1d timeframe identifies overbought/oversold conditions
- Only trade breakouts when 1d Williams %R is NOT extreme (> -20 for longs, < -80 for shorts) 
  to avoid buying tops/selling bottoms in ranging markets
- Volume confirmation (> 1.8x 20-period average) ensures breakout has momentum
- Camarilla R3/S3 levels provide strong support/resistance from prior day's range
- Designed for 4h timeframe targeting 19-50 trades/year (75-200 over 4 years)
- Works in both bull and bear markets by avoiding extreme readings
"""

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
    
    # Get daily data for Camarilla pivots and Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for daily timeframe
    # Based on prior day's OHLC: R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    # where C = (H+L+C)/3 (typical price)
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    camarilla_r3 = typical_price_1d + (range_1d * 1.1 / 4.0)
    camarilla_s3 = typical_price_1d - (range_1d * 1.1 / 4.0)
    
    # Calculate 1d Williams %R(14): (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Values: 0 to -100, where > -20 = overbought, < -80 = oversold
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # Align indicators to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14)  # Williams %R needs 14, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(williams_r_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine breakout conditions with Williams %R filter
        # Long: price breaks above R3, NOT overbought, volume spike
        # Short: price breaks below S3, NOT oversold, volume spike
        price_above_r3 = close[i] > r3_aligned[i]
        price_below_s3 = close[i] < s3_aligned[i]
        
        # Williams %R filter: avoid extremes
        not_overbought = williams_r_aligned[i] > -20  # Not in overbought territory
        not_oversold = williams_r_aligned[i] < -80    # Not in oversold territory
        
        if position == 0:
            # Long conditions: price breaks above R3, not overbought, volume spike
            long_signal = (price_above_r3 and 
                          not_overbought and
                          volume[i] > 1.8 * vol_ma[i])
            
            # Short conditions: price breaks below S3, not oversold, volume spike
            short_signal = (price_below_s3 and 
                           not_oversold and
                           volume[i] > 1.8 * vol_ma[i])
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite level break or Williams %R reaches extreme
            exit_signal = False
            
            if position == 1:
                # Exit long: price falls below S3 or Williams %R becomes overbought
                if (price_below_s3 or 
                    williams_r_aligned[i] <= -20):  # Reached overbought
                    exit_signal = True
            elif position == -1:
                # Exit short: price rises above R3 or Williams %R becomes oversold
                if (price_above_r3 or 
                    williams_r_aligned[i] >= -80):  # Reached oversold
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dWilliamsR_VolumeConfirm"
timeframe = "4h"
leverage = 1.0