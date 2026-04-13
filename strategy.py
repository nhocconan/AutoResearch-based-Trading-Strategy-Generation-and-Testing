#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with 1d Camarilla pivot levels and volume confirmation.
# Long: Price closes above Camarilla R3 level + volume > 1.5x 20-period average + price above 1d EMA50.
# Short: Price closes below Camarilla S3 level + volume > 1.5x 20-period average + price below 1d EMA50.
# Uses 1d for pivot levels and trend filter, 4h for execution with volume confirmation.
# No session filter to allow more trades in different sessions.
# Target: 20-50 total trades over 4 years (5-12/year) for 4h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivot levels and EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each 1d bar
    camarilla_r3 = np.full(len(high_1d), np.nan)
    camarilla_s3 = np.full(len(low_1d), np.nan)
    pivot = np.full(len(close_1d), np.nan)
    
    for i in range(len(close_1d)):
        if i == 0:
            # For first bar, use same values (no previous day)
            camarilla_r3[i] = high_1d[i]
            camarilla_s3[i] = low_1d[i]
            pivot[i] = close_1d[i]
        else:
            # Standard Camarilla formula using previous day's data
            high_y = high_1d[i-1]
            low_y = low_1d[i-1]
            close_y = close_1d[i-1]
            range_y = high_y - low_y
            
            camarilla_r3[i] = close_y + 1.1 * range_y / 2
            camarilla_s3[i] = close_y - 1.1 * range_y / 2
            pivot[i] = (high_y + low_y + close_y) / 3
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # Align 1d Camarilla levels and EMA50 to 4h
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        ema_trend = ema_50_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: close above R3 + above EMA50 + volume confirmation
            if (price > r3 and 
                price > ema_trend and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: close below S3 + below EMA50 + volume confirmation
            elif (price < s3 and 
                  price < ema_trend and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below S3 or below EMA50
            if (price < s3 or
                price < ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price closes above R3 or above EMA50
            if (price > r3 or
                price > ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Camarilla_Pivot_EMA_Volume"
timeframe = "4h"
leverage = 1.0