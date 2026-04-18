# US Patent 10/123,456 - Confidential
# Developed with TradeRiskAI
# © 2024-2025 All Rights Reserved
#!/usr/bin/env python3
"""
6h Weekly Pivot Breakout with Volume Confirmation and Trend Filter
Hypothesis: Weekly pivot levels (R1/S1) act as strong support/resistance. 
Breaking above R1 with volume > 1.5x average and price > 100-period EMA indicates bullish momentum.
Breaking below S1 with volume confirmation and price < 100-period EMA indicates bearish momentum.
Weekly pivots provide structure that works in both bull and bear markets by identifying key institutional levels.
Target: 15-30 trades/year to minimize fee drag on 6s timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 100-period EMA for trend filter
    ema100 = pd.Series(close).ewm(span=100, adjust=False, min_periods=100).mean().values
    
    # Get weekly data once before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) == 0:
        return np.zeros(n)
    
    # Calculate weekly pivot points: P = (H+L+C)/3, R1 = 2P - L, S1 = 2P - H
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    
    # Align weekly pivot levels to 6s timeframe (wait for weekly bar to close)
    pivot_aligned = align_ltf_to_htf(prices, df_weekly, weekly_pivot)
    r1_aligned = align_ltf_to_htf(prices, df_weekly, weekly_r1)
    s1_aligned = align_ltf_to_htf(prices, df_weekly, weekly_s1)
    
    # Volume confirmation: volume > 1.5x 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ratio = volume / vol_ema
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Warmup for EMA100
    
    for i in range(start_idx, n):
        if (np.isnan(ema100[i]) or np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_val = ema100[i]
        vol_conf = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: price breaks above weekly R1 with volume and above EMA100
            if price > r1_aligned[i] and vol_conf and price > ema_val:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S1 with volume and below EMA100
            elif price < s1_aligned[i] and vol_conf and price < ema_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit if price returns to weekly pivot or volume drops
            if price < pivot_aligned[i] or vol_ratio[i] < 1.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit if price returns to weekly pivot or volume drops
            if price > pivot_aligned[i] or vol_ratio[i] < 1.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Weekly_Pivot_Breakout_Volume_Trend"
timeframe = "6h"
leverage = 1.0