#!/usr/bin/env python3
# 1h_4h_1d_camarilla_breakout_v1
# Hypothesis: 1h breakout of daily and 4h Camarilla levels with 4h EMA50 trend filter and volume confirmation.
# Long when price breaks above H4 level (from daily or 4h) with price > 4h EMA50 and volume > 1.5x 10-bar average.
# Short when price breaks below L4 level (from daily or 4h) with price < 4h EMA50 and volume > 1.5x 10-bar average.
# Exit when price returns to opposite Camarilla level (L4 for longs, H4 for shorts).
# Uses 4h and 1d for signal direction, 1h for entry timing. Session filter: 08-20 UTC.
# Position size fixed at 0.20. Target: 60-150 total trades over 4 years (15-37/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_camarilla_breakout_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = np.full(len(close_4h), np.nan)
    if len(close_4h) >= 50:
        ema = close_4h[49]  # Initialize with first 50-period average
        multiplier = 2 / (50 + 1)
        ema_50_4h[49] = ema
        for i in range(50, len(close_4h)):
            ema = (close_4h[i] - ema) * multiplier + ema
            ema_50_4h[i] = ema
    
    # Align 4h EMA50 to 1h timeframe
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate daily Camarilla levels from 1d OHLC
    camarilla_h4_1d = np.full(len(df_1d), np.nan)
    camarilla_l4_1d = np.full(len(df_1d), np.nan)
    for i in range(len(df_1d)):
        c = df_1d['close'].iloc[i]
        h = df_1d['high'].iloc[i]
        l = df_1d['low'].iloc[i]
        camarilla_h4_1d[i] = c + 1.1 * (h - l) / 2
        camarilla_l4_1d[i] = c - 1.1 * (h - l) / 2
    
    # Calculate 4h Camarilla levels from 4h OHLC
    camarilla_h4_4h = np.full(len(df_4h), np.nan)
    camarilla_l4_4h = np.full(len(df_4h), np.nan)
    for i in range(len(df_4h)):
        c = df_4h['close'].iloc[i]
        h = df_4h['high'].iloc[i]
        l = df_4h['low'].iloc[i]
        camarilla_h4_4h[i] = c + 1.1 * (h - l) / 2
        camarilla_l4_4h[i] = c - 1.1 * (h - l) / 2
    
    # Align Camarilla levels to 1h timeframe (use the more recent of daily or 4h)
    camarilla_h4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4_1d)
    camarilla_l4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4_1d)
    camarilla_h4_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h4_4h)
    camarilla_l4_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l4_4h)
    
    # For signal generation, use the tighter (more recent) levels
    # For longs: use the lower of the two H4 levels (earlier breakout)
    # For shorts: use the higher of the two L4 levels (earlier breakout)
    camarilla_h4_combined = np.minimum(camarilla_h4_1d_aligned, camarilla_h4_4h_aligned)
    camarilla_l4_combined = np.maximum(camarilla_l4_1d_aligned, camarilla_l4_4h_aligned)
    
    # Volume confirmation: 10-period average
    vol_ma_10 = np.full(n, np.nan)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 10:
            vol_sum -= volume[i-10]
        if i >= 9:
            vol_ma_10[i] = vol_sum / 10
    
    # Session filter: 08-20 UTC (pre-compute hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(camarilla_h4_combined[i]) or 
            np.isnan(camarilla_l4_combined[i]) or 
            np.isnan(vol_ma_10[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade between 08:00-20:00 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                # Close position outside session
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to or below L4 level
            if close[i] <= camarilla_l4_combined[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price returns to or above H4 level
            if close[i] >= camarilla_h4_combined[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Enter long: price breaks above H4 with trend and volume filters
            if (close[i] > camarilla_h4_combined[i] and 
                close[i] > ema_50_4h_aligned[i] and 
                volume[i] > vol_ma_10[i] * 1.5):
                position = 1
                signals[i] = 0.20
            # Enter short: price breaks below L4 with trend and volume filters
            elif (close[i] < camarilla_l4_combined[i] and 
                  close[i] < ema_50_4h_aligned[i] and 
                  volume[i] > vol_ma_10[i] * 1.5):
                position = -1
                signals[i] = -0.20
    
    return signals