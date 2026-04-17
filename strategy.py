#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe with 12h Camarilla pivot (R2/S2) breakout + 12h volume confirmation + 1d EMA50 trend filter.
Long when price breaks above 12h R2 with volume > 1.8x 20-period 12h average and close > 1d EMA50.
Short when price breaks below 12h S2 with volume > 1.8x 20-period 12h average and close < 1d EMA50.
Exit when price returns to 12h pivot or EMA50 filter fails.
Camarilla R2/S2 represent stronger intraday support/resistance than R1/S1, reducing false breakouts.
Volume confirmation ensures institutional participation. EMA50 filter ensures alignment with daily trend.
Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag. Uses discrete sizing 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla pivots and volume
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h Camarilla levels (R2, S2, pivot)
    # Pivot = (H + L + C) / 3
    pivot_12h = (high_12h + low_12h + close_12h) / 3
    range_12h = high_12h - low_12h
    # R2 = Pivot + Range * 1.1 / 4
    # S2 = Pivot - Range * 1.1 / 4
    r2_12h = pivot_12h + range_12h * 0.275
    s2_12h = pivot_12h - range_12h * 0.275
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50)
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 12h volume 20-period average
    vol_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Align all to 6h
    r2_12h_aligned = align_htf_to_ltf(prices, df_12h, r2_12h)
    s2_12h_aligned = align_htf_to_ltf(prices, df_12h, s2_12h)
    pivot_12h_aligned = align_htf_to_ltf(prices, df_12h, pivot_12h)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    vol_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    volume_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 70  # need enough for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r2_12h_aligned[i]) or np.isnan(s2_12h_aligned[i]) or 
            np.isnan(pivot_12h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma_20_12h_aligned[i]) or np.isnan(volume_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.8x 20-period average
        volume_confirmed = volume_12h_aligned[i] > 1.8 * vol_ma_20_12h_aligned[i]
        
        # EMA50 trend filter: price above/below daily EMA50
        ema_filter_long = close[i] > ema_50_1d_aligned[i]
        ema_filter_short = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above 12h R2 with volume and EMA filter
            if (close[i] > r2_12h_aligned[i] and 
                volume_confirmed and 
                ema_filter_long):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12h S2 with volume and EMA filter
            elif (close[i] < s2_12h_aligned[i] and 
                  volume_confirmed and 
                  ema_filter_short):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below 12h pivot or EMA50 filter fails
            if (close[i] < pivot_12h_aligned[i] or 
                not ema_filter_long):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above 12h pivot or EMA50 filter fails
            if (close[i] > pivot_12h_aligned[i] or 
                not ema_filter_short):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_12hCamarillaR2S2_Volume_EMA50"
timeframe = "6h"
leverage = 1.0