#!/usr/bin/env python3
"""
12h_1d_Camarilla_R3_S3_Breakout_Trend_Volume
Hypothesis: Uses 1d Camarilla pivot levels (R3/S3) for breakout signals on 12h chart.
Requires 1d EMA34 trend filter and volume spike (>2.0x) for confirmation.
Designed to work in both bull and bear markets by following 1d trend direction.
Targets low trade frequency (12-37/year) via 12h timeframe and strict breakout conditions.
"""

name = "12h_1d_Camarilla_R3_S3_Breakout_Trend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels"""
    pivot = (high + low + close) / 3
    range_ = high - low
    r3 = pivot + (range_ * 1.1 / 2)
    s3 = pivot - (range_ * 1.1 / 2)
    return pivot, r3, s3

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d Camarilla for Breakout Levels ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    pivot_1d, r3_1d, s3_1d = calculate_camarilla(
        df_1d['high'].values, df_1d['low'].values, df_1d['close'].values
    )
    
    # Align 1d Camarilla to 12h timeframe
    pivot_1d_12h = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r3_1d_12h = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_12h = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # --- 1d EMA34 for Trend Filter ---
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_12h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # --- Volume Spike Detection (24-period average on 12h) ---
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_1d_12h[i]) or np.isnan(r3_1d_12h[i]) or 
            np.isnan(s3_1d_12h[i]) or np.isnan(ema_34_1d_12h[i]) or
            np.isnan(vol_ratio[i])):
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
            # Long: price breaks above R3 + above 1d EMA34 + volume spike
            if (close[i] > r3_1d_12h[i] and 
                close[i] > ema_34_1d_12h[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 + below 1d EMA34 + volume spike
            elif (close[i] < s3_1d_12h[i] and 
                  close[i] < ema_34_1d_12h[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price returns to pivot or opposite breakout
            if position == 1:
                # Exit long: price breaks below pivot OR breaks below S3
                if close[i] < pivot_1d_12h[i] or close[i] < s3_1d_12h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price breaks above pivot OR breaks above R3
                if close[i] > pivot_1d_12h[i] or close[i] > r3_1d_12h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals