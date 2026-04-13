#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot reversal with volume confirmation and daily trend filter.
# Camarilla levels provide strong intraday support/resistance that work in both bull and bear markets.
# Price rejection at these levels with volume confirmation offers high-probability reversals.
# Daily trend filter ensures trades align with higher timeframe momentum.
# Target: 20-50 total trades over 4 years (5-12/year) for 4h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for multi-timeframe analysis
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    camarilla_H5 = np.full(n, np.nan)
    camarilla_H4 = np.full(n, np.nan)
    camarilla_H3 = np.full(n, np.nan)
    camarilla_L3 = np.full(n, np.nan)
    camarilla_L4 = np.full(n, np.nan)
    camarilla_L5 = np.full(n, np.nan)
    
    for i in range(1, n):
        # Use previous day's data (16 bars back for 4h timeframe)
        prev_idx = max(0, i - 16)
        if prev_idx < len(high):
            # Get daily OHLC for previous day
            day_start = max(0, prev_idx - 15)
            day_high = np.max(high[day_start:prev_idx+1])
            day_low = np.min(low[day_start:prev_idx+1])
            day_close = close[prev_idx]
            
            if day_high > day_low:  # Valid range
                range_val = day_high - day_low
                camarilla_H5[i] = day_close + 1.1 * range_val * 1.1
                camarilla_H4[i] = day_close + 1.1 * range_val * 1.0
                camarilla_H3[i] = day_close + 1.1 * range_val * 0.5
                camarilla_L3[i] = day_close - 1.1 * range_val * 0.5
                camarilla_L4[i] = day_close - 1.1 * range_val * 1.0
                camarilla_L5[i] = day_close - 1.1 * range_val * 1.1
    
    # Calculate average volume (20-period) for volume confirmation
    avg_volume = np.zeros(n)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # Calculate daily trend filter (EMA 21)
    close_1d = df_1d['close'].values
    ema_1d = np.zeros(len(close_1d))
    for i in range(21, len(close_1d)):
        if i == 21:
            ema_1d[i] = np.mean(close_1d[:21])
        else:
            ema_1d[i] = (close_1d[i] * 2/22) + (ema_1d[i-1] * 20/22)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(25, n):
        # Skip if any required data is not ready
        if (np.isnan(camarilla_H3[i]) or np.isnan(camarilla_L3[i]) or 
            np.isnan(avg_volume[i]) or np.isnan(ema_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_trend = ema_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.3x average volume
        volume_confirm = vol > 1.3 * avg_vol
        
        if position == 0:
            # Long: price rejects above L3 level with volume and above daily EMA
            if (price > camarilla_L3[i] and 
                low[i] <= camarilla_L3[i] * 1.001 and  # Touched or slightly below
                volume_confirm and
                price > ema_trend):
                position = 1
                signals[i] = position_size
            # Short: price rejects below H3 level with volume and below daily EMA
            elif (price < camarilla_H3[i] and 
                  high[i] >= camarilla_H3[i] * 0.999 and  # Touched or slightly above
                  volume_confirm and
                  price < ema_trend):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches H3 level or volume drops significantly
            if (price >= camarilla_H3[i] or 
                vol < 0.6 * avg_vol):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price reaches L3 level or volume drops significantly
            if (price <= camarilla_L3[i] or 
                vol < 0.6 * avg_vol):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Camarilla_Pivot_Reversal_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0