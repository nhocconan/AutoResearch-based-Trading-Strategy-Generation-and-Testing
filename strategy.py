#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot levels from 1d timeframe with volume confirmation and trend filter.
# Camarilla levels provide precise support/resistance levels based on previous day's price action.
# In strong trends, price breaks through R4/S4 levels with continuation.
# In ranging markets, price tends to revert from R3/S3 levels.
# Combined with 12h EMA trend filter and volume confirmation to filter false signals.
# Target: 12-37 trades per year (50-150 total over 4 years) for 6h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r3 = np.zeros(len(df_1d))
    camarilla_r4 = np.zeros(len(df_1d))
    camarilla_s3 = np.zeros(len(df_1d))
    camarilla_s4 = np.zeros(len(df_1d))
    
    for i in range(len(df_1d)):
        # Camarilla formulas based on previous day's range
        high_val = high_1d[i]
        low_val = low_1d[i]
        close_val = close_1d[i]
        range_val = high_val - low_val
        
        camarilla_r3[i] = close_val + range_val * 1.1 / 2
        camarilla_r4[i] = close_val + range_val * 1.1
        camarilla_s3[i] = close_val - range_val * 1.1 / 2
        camarilla_s4[i] = close_val - range_val * 1.1
    
    # Align Camarilla levels to 6h timeframe (previous day's levels)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # 12-hour data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # EMA(50) for 12h trend filter
    ema50_12h = np.zeros(len(close_12h))
    ema_multiplier = 2 / (50 + 1)
    ema50_12h[0] = close_12h[0]
    for i in range(1, len(close_12h)):
        ema50_12h[i] = (close_12h[i] - ema50_12h[i-1]) * ema_multiplier + ema50_12h[i-1]
    
    # Align 12h EMA to 6h timeframe
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Average volume (24-period = 12 hours) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(24, n):
        avg_volume[i] = np.mean(volume[i-24:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(24, n):
        # Skip if any required data is not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(ema50_12h_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        camarilla_r3 = camarilla_r3_aligned[i]
        camarilla_r4 = camarilla_r4_aligned[i]
        camarilla_s3 = camarilla_s3_aligned[i]
        camarilla_s4 = camarilla_s4_aligned[i]
        ema_trend = ema50_12h_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: Price breaks above R4 with volume confirmation and above 12h EMA
            if (price > camarilla_r4 and 
                volume_confirm and
                price > ema_trend):
                position = 1
                signals[i] = position_size
            # Short: Price breaks below S4 with volume confirmation and below 12h EMA
            elif (price < camarilla_s4 and 
                  volume_confirm and
                  price < ema_trend):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Price falls below R3 or breaks below 12h EMA
            if (price < camarilla_r3 or
                price < ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Price rises above S3 or breaks above 12h EMA
            if (price > camarilla_s3 or
                price > ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_Camarilla_Pivot_Trend_Volume"
timeframe = "6h"
leverage = 1.0