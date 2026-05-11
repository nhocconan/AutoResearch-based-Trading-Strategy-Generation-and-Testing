#!/usr/bin/env python3
"""
12h_Pivot_Power_Strategy
Hypothesis: Uses 1-week pivot points (weekly high/low/close) for breakout entries on 12h chart,
confirmed by 1-day EMA200 trend and volume spikes. Designed for low trade frequency by requiring
confluence of price breaking key weekly pivot levels, long-term trend alignment, and volume confirmation.
Works in bull markets by buying breakouts above weekly resistance and in bear markets by selling
breakouts below weekly support, while avoiding counter-trend trades.
"""

name = "12h_Pivot_Power_Strategy"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1-week OHLC for Weekly Pivot Points ---
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate pivot points using previous 1-week's OHLC
    prev_high_1w = df_1w['high'].values
    prev_low_1w = df_1w['low'].values
    prev_close_1w = df_1w['close'].values
    
    # Weekly pivot point (standard formula)
    weekly_pivot = (prev_high_1w + prev_low_1w + prev_close_1w) / 3.0
    weekly_range = prev_high_1w - prev_low_1w
    
    # Key levels: Weekly R1 and S1 (more frequently tested than R3/S3)
    weekly_R1 = 2 * weekly_pivot - prev_low_1w
    weekly_S1 = 2 * weekly_pivot - prev_high_1w
    
    # Align to 12h timeframe
    weekly_R1_12h = align_htf_to_ltf(prices, df_1w, weekly_R1)
    weekly_S1_12h = align_htf_to_ltf(prices, df_1w, weekly_S1)
    
    # --- 1-day EMA200 Trend Filter (long-term trend) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    ema_200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_12h = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # --- Volume Spike Detection (30-period average on 12h) ---
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for EMA200 and weekly pivot calculation)
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN (first few bars)
        if (np.isnan(weekly_R1_12h[i]) or np.isnan(weekly_S1_12h[i]) or 
            np.isnan(ema_200_12h[i]) or np.isnan(vol_ratio[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation threshold
        volume_spike = vol_ratio[i] > 2.0
        
        if position == 0:
            # Long: price breaks above weekly R1 with volume, above EMA200 (bullish bias)
            if (close[i] > weekly_R1_12h[i] and 
                volume_spike and 
                close[i] > ema_200_12h[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S1 with volume, below EMA200 (bearish bias)
            elif (close[i] < weekly_S1_12h[i] and 
                  volume_spike and 
                  close[i] < ema_200_12h[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite breakout or loss of momentum
            if position == 1:
                # Exit long: price breaks below weekly S1 (reversal signal)
                if close[i] < weekly_S1_12h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price breaks above weekly R1 (reversal signal)
                if close[i] > weekly_R1_12h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals