#!/usr/bin/env python3
"""
6h_camarilla_pivot_1d_ema_volume_v2
Hypothesis: Use daily Camarilla pivot levels (R3/S3, R4/S4) from 1d timeframe for mean reversion on 6s.
Long when price touches S3 with EMA50 > EMA200 on 1d (bullish bias) and volume > 1.5x average.
Short when price touches R3 with EMA50 < EMA200 on 1d (bearish bias) and volume > 1.5x average.
Exit when price crosses the daily pivot (mean reversion complete) or volume drops.
Targets 12-37 trades/year to minimize fee drift.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_1d_ema_volume_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # EMA crossover on 6h for entry confirmation (fast=12, slow=26)
    close_s = pd.Series(close)
    ema12 = close_s.ewm(span=12, min_periods=12, adjust=False).mean().values
    ema26 = close_s.ewm(span=26, min_periods=26, adjust=False).mean().values
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get daily data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily OHLC for Camarilla pivot
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    
    # Camarilla levels: R3, R4, S3, S4
    # R4 = close + ((high - low) * 1.1 / 2)
    # R3 = close + ((high - low) * 1.1 / 4)
    # S3 = close - ((high - low) * 1.1 / 4)
    # S4 = close - ((high - low) * 1.1 / 2)
    # Pivot = (high + low + close) / 3
    pivot = (d_high + d_low + d_close) / 3
    range_hl = d_high - d_low
    r3 = pivot + (range_hl * 1.1 / 4)
    r4 = pivot + (range_hl * 1.1 / 2)
    s3 = pivot - (range_hl * 1.1 / 4)
    s4 = pivot - (range_hl * 1.1 / 2)
    
    # Align to 6h timeframe (shifted by 1 day)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    pivot_6h = align_htf_to_ltf(prices, df_1d, pivot)
    
    # EMA50 and EMA200 on daily close for trend filter
    d_close_s = pd.Series(d_close)
    ema50_1d = d_close_s.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema200_1d = d_close_s.ewm(span=200, min_periods=200, adjust=False).mean().values
    ema50_1d_6h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema200_1d_6h = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(ema12[i]) or np.isnan(ema26[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0 or
            np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or
            np.isnan(pivot_6h[i]) or
            np.isnan(ema50_1d_6h[i]) or np.isnan(ema200_1d_6h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter from 1d
        trend_up = ema50_1d_6h[i] > ema200_1d_6h[i]
        trend_down = ema50_1d_6h[i] < ema200_1d_6h[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit when price crosses above pivot (mean reversion complete)
            if close[i] > pivot_6h[i]:
                exit_long = True
            # Exit when volume drops below average
            elif volume[i] < vol_ma[i]:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit when price crosses below pivot (mean reversion complete)
            if close[i] < pivot_6h[i]:
                exit_short = True
            # Exit when volume drops below average
            elif volume[i] < vol_ma[i]:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price at or below S3, bullish trend, volume confirmation
            long_entry = (close[i] <= s3_6h[i]) and trend_up and vol_confirm
            
            # Short entry: price at or above R3, bearish trend, volume confirmation
            short_entry = (close[i] >= r3_6h[i]) and trend_down and vol_confirm
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals