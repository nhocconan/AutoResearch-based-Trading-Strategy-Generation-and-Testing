#!/usr/bin/env python3
"""
4h_1d_TRIX_Volume_Spike_v1
Hypothesis: 4h TRIX crossing above/below zero with 1d volume spike (>1.5x average) and price above/below 1d EMA(50) for trend filter. 
TRIX captures momentum, volume spike confirms institutional interest, EMA filter ensures trend alignment. 
Designed for low trade frequency (20-50/year) by requiring momentum confirmation, volume surge, and trend alignment. 
Works in bull/bear via EMA trend filter and mean-reversion exit when TRIX crosses zero in opposite direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_TRIX_Volume_Spike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY DATA ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Daily EMA(50) for trend filter
    if len(close_1d) >= 50:
        ema_50_1d = np.zeros_like(close_1d)
        ema_50_1d[0] = close_1d[0]
        alpha = 2.0 / (50 + 1)
        for i in range(1, len(close_1d)):
            ema_50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_50_1d[i-1]
    else:
        ema_50_1d = np.full_like(close_1d, np.nan)
    
    # Daily volume average (20-period)
    vol_avg_1d = np.zeros(len(volume_1d))
    vol_sum = 0.0
    vol_count = 0
    for i in range(len(volume_1d)):
        vol_sum += volume_1d[i]
        vol_count += 1
        if i >= 20:
            vol_sum -= volume_1d[i-20]
            vol_count -= 1
        if vol_count > 0:
            vol_avg_1d[i] = vol_sum / vol_count
        else:
            vol_avg_1d[i] = 0.0
    
    # Calculate TRIX (1-period ROC of triple-smoothed EMA)
    # TRIX = 100 * (EMA3 - EMA3_prev) / EMA3_prev
    if len(close) >= 15:
        # First EMA(12)
        ema1 = np.zeros(n)
        ema1[0] = close[0]
        alpha1 = 2.0 / (12 + 1)
        for i in range(1, n):
            ema1[i] = alpha1 * close[i] + (1 - alpha1) * ema1[i-1]
        
        # Second EMA(12) of first EMA
        ema2 = np.zeros(n)
        ema2[0] = ema1[0]
        alpha2 = 2.0 / (12 + 1)
        for i in range(1, n):
            ema2[i] = alpha2 * ema1[i] + (1 - alpha2) * ema2[i-1]
        
        # Third EMA(12) of second EMA
        ema3 = np.zeros(n)
        ema3[0] = ema2[0]
        alpha3 = 2.0 / (12 + 1)
        for i in range(1, n):
            ema3[i] = alpha3 * ema2[i] + (1 - alpha3) * ema3[i-1]
        
        # TRIX calculation
        trix = np.zeros(n)
        trix[0] = 0.0
        for i in range(1, n):
            if ema3[i-1] != 0:
                trix[i] = 100 * (ema3[i] - ema3[i-1]) / ema3[i-1]
            else:
                trix[i] = 0.0
    else:
        trix = np.zeros(n)
    
    # Align daily data to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # Volume spike detection (current 4h volume > 1.5x daily average volume scaled to 4h)
    # Approximate: daily volume / 6 = average 4h volume (since 6*4h = 24h)
    vol_spike = volume > (1.5 * vol_avg_1d_aligned / 6.0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_avg_1d_aligned[i]) or 
            np.isnan(trix[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: price above/below daily EMA(50)
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        
        # TRIX signals: crossing zero
        trix_cross_up = trix[i] > 0 and trix[i-1] <= 0
        trix_cross_down = trix[i] < 0 and trix[i-1] >= 0
        
        # Entry conditions with volume spike and trend filter
        long_entry = trix_cross_up and vol_spike[i] and price_above_ema
        short_entry = trix_cross_down and vol_spike[i] and price_below_ema
        
        # Exit when TRIX crosses zero in opposite direction (mean reversion)
        exit_long = trix_cross_down
        exit_short = trix_cross_up
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals