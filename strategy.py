#!/usr/bin/env python3
# 1d_Camarilla_R3S3_Breakout_1wTrend_Volume
# Hypothesis: Breakout of weekly Camarilla R3/S3 levels on daily chart with confirmation from weekly trend and volume spike.
# Uses 1d timeframe to reduce trade frequency and filter noise. Weekly trend ensures alignment with higher timeframe momentum.
# Volume spike confirms breakout strength. Designed for low trade frequency (<25/year) to minimize fee drag in bear markets.

name = "1d_Camarilla_R3S3_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Weekly Data (loaded ONCE) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # === Weekly Camarilla Pivot Levels (R3, S3) ===
    pivot = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    r3 = pivot + (range_1w * 1.1 / 2)
    s3 = pivot - (range_1w * 1.1 / 2)
    
    # Align weekly levels to daily
    r3_1d = align_htf_to_ltf(prices, df_1w, r3)
    s3_1d = align_htf_to_ltf(prices, df_1w, s3)
    
    # === Weekly EMA34 Trend Filter ===
    ema34_1w = pd.Series(close_1w).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema34_1w_1d = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # === Volume Spike Filter (20-period EMA) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 1.5  # Require 1.5x average volume
    
    # === Signal Parameters ===
    position_size = 0.25  # 25% of capital per trade
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    holding_bars = 0
    
    # Start after warmup (covers EMA34)
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_1d[i]) or np.isnan(s3_1d[i]) or 
            np.isnan(ema34_1w_1d[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                holding_bars = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Break above R3 + above weekly EMA34 + volume spike
            if (close[i] > r3_1d[i] and 
                close[i] > ema34_1w_1d[i] and 
                volume_ok[i]):
                signals[i] = position_size
                position = 1
                holding_bars = 0
            # Short: Break below S3 + below weekly EMA34 + volume spike
            elif (close[i] < s3_1d[i] and 
                  close[i] < ema34_1w_1d[i] and 
                  volume_ok[i]):
                signals[i] = -position_size
                position = -1
                holding_bars = 0
        else:
            # Enforce minimum holding period (5 days)
            holding_bars += 1
            if holding_bars < 5:
                signals[i] = position_size if position == 1 else -position_size
                continue
            
            # Exit: Price closes below/above opposite level
            if position == 1:
                if close[i] < s3_1d[i]:
                    signals[i] = 0.0
                    position = 0
                    holding_bars = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                if close[i] > r3_1d[i]:
                    signals[i] = 0.0
                    position = 0
                    holding_bars = 0
                else:
                    signals[i] = -position_size
    
    return signals