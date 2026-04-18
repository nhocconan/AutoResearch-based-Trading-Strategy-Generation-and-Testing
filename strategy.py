#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1wPivot_R1S1_Breakout_VolumeTrendFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for pivot and trend
    df_1w = get_htf_data(prices, '1w')
    
    # Previous weekly OHLC
    prev_high_w = df_1w['high'].shift(1).values
    prev_low_w = df_1w['low'].shift(1).values
    prev_close_w = df_1w['close'].shift(1).values
    
    # Weekly pivot and R1/S1
    pivot_w = (prev_high_w + prev_low_w + prev_close_w) / 3
    range_w = prev_high_w - prev_low_w
    R1_w = pivot_w + range_w
    S1_w = pivot_w - range_w
    
    # Weekly EMA34 for trend filter
    ema34_w = pd.Series(prev_close_w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align to 4h
    R1_w_aligned = align_htf_to_ltf(prices, df_1w, R1_w)
    S1_w_aligned = align_htf_to_ltf(prices, df_1w, S1_w)
    ema34_w_aligned = align_htf_to_ltf(prices, df_1w, ema34_w)
    
    # Volume filter: current volume > 2.5 * 20-period average (20 * 4h = 5 days)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(R1_w_aligned[i]) or np.isnan(S1_w_aligned[i]) or
            np.isnan(ema34_w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        R1_val = R1_w_aligned[i]
        S1_val = S1_w_aligned[i]
        ema34_val = ema34_w_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Long: break above R1 with volume and bullish trend (close > weekly EMA34)
            if close_val > R1_val and vol_filter and close_val > ema34_val:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume and bearish trend (close < weekly EMA34)
            elif close_val < S1_val and vol_filter and close_val < ema34_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls back to S1
            if close_val <= S1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises back to R1
            if close_val >= R1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals