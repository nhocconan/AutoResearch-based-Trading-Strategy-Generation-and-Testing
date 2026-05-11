#/usr/bin/env python3
"""
12h_1d_Camarilla_R1_S1_Breakout_TrendFilter_Volume_v2
Hypothesis: Uses Camarilla pivot levels from 1d timeframe (R1/S1) for breakout entries on 12h chart,
confirmed by 1d EMA21 trend and volume spikes. Designed for low trade frequency by requiring confluence of
price breaking key 1d pivot levels, trend alignment, and volume confirmation. Works in bull and bear
markets by following intermediate-term trend from 1d timeframe.
"""

name = "12h_1d_Camarilla_R1_S1_Breakout_TrendFilter_Volume_v2"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d OHLCV for Camarilla Pivot Levels ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate pivot points using previous 1d's OHLC
    prev_high_1d = df_1d['high'].values
    prev_low_1d = df_1d['low'].values
    prev_close_1d = df_1d['close'].values
    
    pivot_1d = (prev_high_1d + prev_low_1d + prev_close_1d) / 3.0
    range_val_1d = prev_high_1d - prev_low_1d
    
    # Camarilla levels (R1 and S1)
    R1_1d = pivot_1d + (range_val_1d * 1.1 / 12)
    S1_1d = pivot_1d - (range_val_1d * 1.1 / 12)
    
    # Align to 12h timeframe
    R1_12h = align_htf_to_ltf(prices, df_1d, R1_1d)
    S1_12h = align_htf_to_ltf(prices, df_1d, S1_1d)
    
    # --- 1d EMA21 Trend Filter ---
    ema_21_1d = pd.Series(df_1d['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_12h = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    
    # --- Volume Spike Detection (12-period average on 12h) ---
    vol_ma = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for EMA21 and pivot calculation)
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN (first few bars)
        if (np.isnan(R1_12h[i]) or np.isnan(S1_12h[i]) or 
            np.isnan(ema_21_12h[i]) or np.isnan(vol_ratio[i])):
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
            if (close[i] > R1_12h[i] and 
                volume_spike and 
                close[i] > ema_21_12h[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume, below EMA21
            elif (close[i] < S1_12h[i] and 
                  volume_spike and 
                  close[i] < ema_21_12h[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite breakout or loss of momentum
            if position == 1:
                # Exit long: price breaks below S1 (reversal signal)
                if close[i] < S1_12h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price breaks above R1 (reversal signal)
                if close[i] > R1_12h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals