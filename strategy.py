#!/usr/bin/env python3
# 4h_Camarilla_R3S3_Breakout_12hEMA50_Trend_Volume
# Hypothesis: Breakout of 1-day Camarilla R3/S3 levels with 12-hour EMA50 trend filter and volume confirmation.
# Uses 12h EMA50 to filter direction (long above, short below) to reduce false breakouts and capture trend context.
# Targets 20-35 trades/year by requiring volume spike (>2x 20-period EMA) and strict breakout conditions.
# Designed for BTC/ETH resilience in bull/bear markets via trend alignment and volume filters.

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_Trend_Volume"
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
    
    # === 12h EMA50 Trend Filter ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_12h_4h = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # === Volume Spike Filter (20-period EMA) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 2.0  # Require 2x average volume for confirmation
    
    # === Signal Parameters ===
    position_size = 0.25  # 25% of capital per trade
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    holding_bars = 0
    
    # Start after warmup (covers EMA50)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_4h[i]) or np.isnan(s3_4h[i]) or 
            np.isnan(ema50_12h_4h[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                holding_bars = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Break above R3 (both open and close) + above 12h EMA50 + volume spike
            if (open_price[i] > r3_4h[i] and close[i] > r3_4h[i] and 
                close[i] > ema50_12h_4h[i] and volume_ok[i]):
                signals[i] = position_size
                position = 1
                holding_bars = 0
            # Short: Break below S3 (both open and close) + below 12h EMA50 + volume spike
            elif (open_price[i] < s3_4h[i] and close[i] < s3_4h[i] and 
                  close[i] < ema50_12h_4h[i] and volume_ok[i]):
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