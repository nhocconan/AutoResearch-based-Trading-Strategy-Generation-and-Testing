#!/usr/bin/env python3
# 1h_Camarilla_R3S3_Breakout_4hTrend_Volume_Confirm
# Hypothesis: Breakout of 4-hour Camarilla R3/S3 levels on 1h chart with confirmation from 4-hour EMA34 trend and volume spike.
# Uses 4h for trend direction and 1h for entry timing to avoid overtrading. Targets 15-37 trades/year.
# Works in both bull and bear markets by requiring trend alignment and volume confirmation.

name = "1h_Camarilla_R3S3_Breakout_4hTrend_Volume_Confirm"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 80:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 4h Data (loaded ONCE) ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # === 4h Camarilla Pivot Levels (R3, S3) ===
    pivot = (high_4h + low_4h + close_4h) / 3
    range_4h = high_4h - low_4h
    r3 = pivot + (range_4h * 1.1 / 2)
    s3 = pivot - (range_4h * 1.1 / 2)
    
    # Align 4h levels to 1h
    r3_1h = align_htf_to_ltf(prices, df_4h, r3)
    s3_1h = align_htf_to_ltf(prices, df_4h, s3)
    
    # === 4h EMA34 Trend Filter ===
    ema34_4h = pd.Series(close_4h).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema34_4h_1h = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # === Volume Spike Filter (20-period EMA) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 1.5  # Require 1.5x average volume
    
    # === Session Filter: 08-20 UTC ===
    # Pre-compute hours from datetime index
    hours = prices.index.hour
    session_ok = (hours >= 8) & (hours <= 20)
    
    # === Signal Parameters ===
    position_size = 0.20  # 20% of capital per trade
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    holding_bars = 0
    
    # Start after warmup (covers EMA34)
    start_idx = 80
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_1h[i]) or np.isnan(s3_1h[i]) or 
            np.isnan(ema34_4h_1h[i]) or np.isnan(volume_ok[i]) or 
            np.isnan(session_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                holding_bars = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Check session filter
            if not session_ok[i]:
                signals[i] = 0.0
                continue
                
            # Long: Break above R3 + above 4h EMA34 + volume spike
            if (close[i] > r3_1h[i] and 
                close[i] > ema34_4h_1h[i] and 
                volume_ok[i]):
                signals[i] = position_size
                position = 1
                holding_bars = 0
            # Short: Break below S3 + below 4h EMA34 + volume spike
            elif (close[i] < s3_1h[i] and 
                  close[i] < ema34_4h_1h[i] and 
                  volume_ok[i]):
                signals[i] = -position_size
                position = -1
                holding_bars = 0
        else:
            # Enforce minimum holding period (6 bars)
            holding_bars += 1
            if holding_bars < 6:
                signals[i] = position_size if position == 1 else -position_size
                continue
            
            # Exit: Price closes below/above opposite level
            if position == 1:
                if close[i] < s3_1h[i]:
                    signals[i] = 0.0
                    position = 0
                    holding_bars = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                if close[i] > r3_1h[i]:
                    signals[i] = 0.0
                    position = 0
                    holding_bars = 0
                else:
                    signals[i] = -position_size
    
    return signals