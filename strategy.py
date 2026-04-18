#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_WeeklyPivot_R1_S1_Breakout_Volume_TrendFilter"
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
    
    # Load 1w data for Weekly Pivot R1/S1
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Weekly Pivot R1 and S1 from previous weekly bar
    prev_close = df_1w['close'].shift(1).values
    prev_high = df_1w['high'].shift(1).values
    prev_low = df_1w['low'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    R1 = pivot + range_hl
    S1 = pivot - range_hl
    
    # Align R1/S1 to 4h (wait for weekly close)
    R1_aligned = align_htf_to_ltf(prices, df_1w, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1w, S1)
    
    # Volume filter: current volume > 1.8 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.8 * vol_ma_20)
    
    # Trend filter: 4h EMA50
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(ema_50[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        R1_val = R1_aligned[i]
        S1_val = S1_aligned[i]
        ema_50_val = ema_50[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Long: break above R1 with volume confirmation and price above EMA50
            if close_val > R1_val and vol_filter and (close_val > ema_50_val):
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume confirmation and price below EMA50
            elif close_val < S1_val and vol_filter and (close_val < ema_50_val):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls back below S1 or EMA50 turns bearish
            if close_val < S1_val or (close_val < ema_50_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises back above R1 or EMA50 turns bullish
            if close_val > R1_val or (close_val > ema_50_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals