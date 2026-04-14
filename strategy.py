#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Weekly Camarilla Pivot Breakout with 1-day Volume Confirmation
# Long when price breaks above weekly R4 level AND volume > 2x daily average
# Short when price breaks below weekly S4 level AND volume > 2x daily average
# Exit when price returns to weekly pivot (R3/S3) or closes back inside weekly H-L range
# Uses weekly structure for direction and daily volume for confirmation to catch strong trending moves
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee impact

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for Camarilla pivots
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly Camarilla levels from prior week's OHLC
    # H, L, C from previous week (already completed due to get_htf_data alignment)
    H = df_1w['high'].values
    L = df_1w['low'].values
    C = df_1w['close'].values
    
    # Camarilla formulas
    R4 = C + (H - L) * 1.1 / 2
    R3 = C + (H - L) * 1.1 / 4
    S3 = C - (H - L) * 1.1 / 4
    S4 = C - (H - L) * 1.1 / 2
    
    # Align weekly levels to 6h timeframe (wait for weekly bar to close)
    R4_6h = align_htf_to_ltf(prices, df_1w, R4)
    R3_6h = align_htf_to_ltf(prices, df_1w, R3)
    S3_6h = align_htf_to_ltf(prices, df_1w, S3)
    S4_6h = align_htf_to_ltf(prices, df_1w, S4)
    
    # Weekly high-low range for exit condition
    WHL_range = H - L
    weekly_mid = (H + L) / 2
    WHL_range_6h = align_htf_to_ltf(prices, df_1w, WHL_range)
    weekly_mid_6h = align_htf_to_ltf(prices, df_1w, weekly_mid)
    
    # Load daily data ONCE before loop for volume average
    df_1d = get_htf_data(prices, '1d')
    
    # Daily volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_avg_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_6h = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 30
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(R4_6h[i]) or np.isnan(S4_6h[i]) or 
            np.isnan(vol_avg_20_6h[i]) or np.isnan(WHL_range_6h[i]) or 
            np.isnan(weekly_mid_6h[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg_20_6h[i] * 2.0  # Require 2x daily average volume
        
        if position == 0:
            # Long setup: break above weekly R4 with volume confirmation
            if price > R4_6h[i] and vol > vol_threshold:
                position = 1
                signals[i] = position_size
            # Short setup: break below weekly S4 with volume confirmation
            elif price < S4_6h[i] and vol > vol_threshold:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to weekly S3 or below weekly midpoint
            if price < S3_6h[i] or price < weekly_mid_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to weekly R3 or above weekly midpoint
            if price > R3_6h[i] or price > weekly_mid_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_WeeklyCamarilla_VolumeBreakout"
timeframe = "6h"
leverage = 1.0