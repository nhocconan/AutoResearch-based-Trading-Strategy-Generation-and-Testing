#!/usr/bin/env python3
"""
6h_Weekly_RangeBreakout_4hTrend
Hypothesis: Price often consolidates within weekly ranges before breaking out.
Enter on weekly high/low breakouts with 4h trend alignment and volume confirmation.
Weekly structure provides meaningful support/resistance that works in both bull/bear markets.
Targets 15-30 trades/year by requiring confluence of weekly breakout, trend, and volume.
"""

name = "6h_Weekly_RangeBreakout_4hTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for range
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Weekly High and Low ---
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    
    # Align weekly levels to 6s
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    # --- 4h EMA20 Trend Filter ---
    close_4h = df_4h['close'].values
    ema_20 = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_4h, ema_20)
    
    # --- Volume Spike Detection (20-period average) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or
            np.isnan(ema_20_aligned[i]) or np.isnan(vol_ratio[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation threshold
        volume_spike = vol_ratio[i] > 1.8
        
        if position == 0:
            # Long: price breaks above weekly high with volume, above 4h EMA20
            if (close[i] > weekly_high_aligned[i] and 
                volume_spike and 
                close[i] > ema_20_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly low with volume, below 4h EMA20
            elif (close[i] < weekly_low_aligned[i] and 
                  volume_spike and 
                  close[i] < ema_20_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite weekly breakout or loss of momentum
            if position == 1:
                # Exit long: price breaks below weekly low
                if close[i] < weekly_low_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price breaks above weekly high
                if close[i] > weekly_high_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals