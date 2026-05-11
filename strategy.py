#!/usr/bin/env python3
"""
1d_1w_Camarilla_R1_S1_Breakout_TrendFilter_Volume
Hypothesis: Uses weekly pivot levels (R1/S1) from 1w timeframe for breakout entries on 1d chart,
confirmed by 1w EMA21 trend and volume spikes. Weekly timeframe reduces whipsaw, provides
stronger trend context, and lowers trade frequency to avoid fee drag. Works in both bull and
bear markets by following the higher-timeframe trend.
"""

name = "1d_1w_Camarilla_R1_S1_Breakout_TrendFilter_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # 1d OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1w OHLCV for Camarilla Pivot Levels ---
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate pivot points using previous 1w's OHLC
    prev_high_1w = df_1w['high'].values
    prev_low_1w = df_1w['low'].values
    prev_close_1w = df_1w['close'].values
    
    pivot_1w = (prev_high_1w + prev_low_1w + prev_close_1w) / 3.0
    range_val_1w = prev_high_1w - prev_low_1w
    
    # Camarilla levels (R1 and S1)
    R1_1w = pivot_1w + (range_val_1w * 1.1 / 12)
    S1_1w = pivot_1w - (range_val_1w * 1.1 / 12)
    
    # Align to 1d timeframe
    R1_1d = align_htf_to_ltf(prices, df_1w, R1_1w)
    S1_1d = align_htf_to_ltf(prices, df_1w, S1_1w)
    
    # --- 1w EMA21 Trend Filter ---
    ema_21_1w = pd.Series(df_1w['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1d = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # --- Volume Spike Detection (12-period average on 1d) ---
    vol_ma = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for EMA21 and pivot calculation)
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN (first few bars)
        if (np.isnan(R1_1d[i]) or np.isnan(S1_1d[i]) or 
            np.isnan(ema_21_1d[i]) or np.isnan(vol_ratio[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation threshold
        volume_spike = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: price breaks above R1 with volume, above EMA21
            if (close[i] > R1_1d[i] and 
                volume_spike and 
                close[i] > ema_21_1d[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume, below EMA21
            elif (close[i] < S1_1d[i] and 
                  volume_spike and 
                  close[i] < ema_21_1d[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite breakout or loss of momentum
            if position == 1:
                # Exit long: price breaks below S1 (reversal signal)
                if close[i] < S1_1d[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price breaks above R1 (reversal signal)
                if close[i] > R1_1d[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals