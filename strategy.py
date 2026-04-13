#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with 1d Camarilla pivot levels and volume confirmation.
# Long: Price breaks above 1d Camarilla R4 level + volume > 1.5x average volume (20-period).
# Short: Price breaks below 1d Camarilla S4 level + volume > 1.5x average volume.
# Uses Camarilla pivots from daily timeframe for key support/resistance levels.
# Volume confirmation reduces false breakouts. Position size 0.25.
# Session filter: 08-20 UTC to avoid low-liquidity hours.
# Target: 20-50 total trades over 4 years (5-12.5/year) for 4h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Session filter: 08-20 UTC (pre-compute hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels
    camarilla_r4 = np.full(len(close_1d), np.nan)
    camarilla_r3 = np.full(len(close_1d), np.nan)
    camarilla_r2 = np.full(len(close_1d), np.nan)
    camarilla_r1 = np.full(len(close_1d), np.nan)
    camarilla_s1 = np.full(len(close_1d), np.nan)
    camarilla_s2 = np.full(len(close_1d), np.nan)
    camarilla_s3 = np.full(len(close_1d), np.nan)
    camarilla_s4 = np.full(len(close_1d), np.nan)
    pivot = np.full(len(close_1d), np.nan)
    
    for i in range(len(close_1d)):
        if i < 1:  # Need previous day's data
            continue
        # Use previous day's OHLC to calculate today's pivots
        high_prev = high_1d[i-1]
        low_prev = low_1d[i-1]
        close_prev = close_1d[i-1]
        
        pivot[i] = (high_prev + low_prev + close_prev) / 3
        range_prev = high_prev - low_prev
        
        camarilla_r1[i] = close_prev + range_prev * 1.1 / 12
        camarilla_s1[i] = close_prev - range_prev * 1.1 / 12
        camarilla_r2[i] = close_prev + range_prev * 1.1 / 6
        camarilla_s2[i] = close_prev - range_prev * 1.1 / 6
        camarilla_r3[i] = close_prev + range_prev * 1.1 / 4
        camarilla_s3[i] = close_prev - range_prev * 1.1 / 4
        camarilla_r4[i] = close_prev + range_prev * 1.1 / 2
        camarilla_s4[i] = close_prev - range_prev * 1.1 / 2
    
    # Average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # Align 1d Camarilla levels to 4h
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_r2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r2)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_s2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s2)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        if not (8 <= hour <= 20):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        r4 = camarilla_r4_aligned[i]
        s4 = camarilla_s4_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: break above Camarilla R4 + volume confirmation
            if (price > r4 and volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: break below Camarilla S4 + volume confirmation
            elif (price < s4 and volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below Camarilla R3 (take profit) or S4 (stop)
            if (price < camarilla_r3_aligned[i] or
                price < camarilla_s4_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price closes above Camarilla S3 (take profit) or R4 (stop)
            if (price > camarilla_s3_aligned[i] or
                price > camarilla_r4_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Camarilla_Pivot_Volume"
timeframe = "4h"
leverage = 1.0