#!/usr/bin/env python3
# 4h_1d_camarilla_breakout_v1
# Hypothesis: 4-hour Camarilla pivot levels from daily timeframe with volume confirmation and trend filter.
# Long when price closes above Camarilla H4 resistance with price > daily EMA50 and volume > 1.5x average.
# Short when price closes below Camarilla L4 support with price < daily EMA50 and volume > 1.5x average.
# Uses Camarilla levels (based on previous day's range) for precise support/resistance levels.
# Works in both bull and bear markets by following the daily trend filter to avoid counter-trend trades.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        # Simple EMA calculation with proper initialization
        ema = np.mean(close_1d[:50])
        ema_50_1d[49] = ema
        multiplier = 2 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema = (close_1d[i] - ema) * multiplier + ema
            ema_50_1d[i] = ema
    
    # Align daily EMA50 to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Camarilla: H4 = C + 1.5*(H-L), L4 = C - 1.5*(H-L) where C,H,L are from previous day
    camarilla_h4 = np.full(len(close_1d), np.nan)
    camarilla_l4 = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):
        prev_high = df_1d['high'].iloc[i-1]
        prev_low = df_1d['low'].iloc[i-1]
        prev_close = df_1d['close'].iloc[i-1]
        camarilla_h4[i] = prev_close + 1.5 * (prev_high - prev_low)
        camarilla_l4[i] = prev_close - 1.5 * (prev_high - prev_low)
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Volume confirmation: 20-period average
    vol_ma_20 = np.full(n, np.nan)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to or below Camarilla L4
            if close[i] <= camarilla_l4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to or above Camarilla H4
            if close[i] >= camarilla_h4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price closes above Camarilla H4 with trend and volume filters
            if (close[i] > camarilla_h4_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume[i] > vol_ma_20[i] * 1.5):
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below Camarilla L4 with trend and volume filters
            elif (close[i] < camarilla_l4_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume[i] > vol_ma_20[i] * 1.5):
                position = -1
                signals[i] = -0.25
    
    return signals