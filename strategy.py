#!/usr/bin/env python3
# 4h_Camarilla_R3S3_Breakout_1dTrend_Volume_Confirm_v4
# Hypothesis: Breakout of 1-day Camarilla R3/S3 levels with 1-day EMA34 trend filter and volume confirmation.
# Uses double confirmation (open and close beyond level) to reduce whipsaws, enforces minimum holding period,
# and targets 25-35 trades/year. Designed for BTC/ETH resilience via trend alignment and volume filters.
# Focuses on stronger breakouts (R3/S3) to improve signal quality and reduce frequency.

name = "4h_Camarilla_R3S3_Breakout_1dTrend_Volume_Confirm_v4"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    open_price = prices['open'].values
    volume = prices['volume'].values
    
    # === 1d Data (loaded ONCE) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d Camarilla Pivot Levels (R3, S3) ===
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r3 = pivot + (range_1d * 1.1 / 2)
    s3 = pivot - (range_1d * 1.1 / 2)
    
    # Align 1d levels to 4h
    r3_4h = align_htf_to_ltf(prices, df_1d, r3)
    s3_4h = align_htf_to_ltf(prices, df_1d, s3)
    
    # === 1d EMA34 Trend Filter ===
    ema34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema34_1d_4h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === Volume Spike Filter (20-period EMA) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 1.5  # Require 1.5x average volume
    
    # === Signal Parameters ===
    position_size = 0.25  # 25% of capital per trade
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    holding_bars = 0
    
    # Start after warmup (covers EMA34)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_4h[i]) or np.isnan(s3_4h[i]) or 
            np.isnan(ema34_1d_4h[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                holding_bars = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Break above R3 (both open and close) + above 1d EMA34 + volume spike
            if (open_price[i] > r3_4h[i] and close[i] > r3_4h[i] and 
                close[i] > ema34_1d_4h[i] and volume_ok[i]):
                signals[i] = position_size
                position = 1
                holding_bars = 0
            # Short: Break below S3 (both open and close) + below 1d EMA34 + volume spike
            elif (open_price[i] < s3_4h[i] and close[i] < s3_4h[i] and 
                  close[i] < ema34_1d_4h[i] and volume_ok[i]):
                signals[i] = -position_size
                position = -1
                holding_bars = 0
        else:
            # Enforce minimum holding period (12 bars)
            holding_bars += 1
            if holding_bars < 12:
                signals[i] = position_size if position == 1 else -position_size
                continue
            
            # Exit: Price closes below/above opposite level
            if position == 1:
                if close[i] < s3_4h[i]:
                    signals[i] = 0.0
                    position = 0
                    holding_bars = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                if close[i] > r3_4h[i]:
                    signals[i] = 0.0
                    position = 0
                    holding_bars = 0
                else:
                    signals[i] = -position_size
    
    return signals