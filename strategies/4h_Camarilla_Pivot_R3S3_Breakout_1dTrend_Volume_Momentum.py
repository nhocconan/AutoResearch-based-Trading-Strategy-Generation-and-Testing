#!/usr/bin/env python3
# 4h_Camarilla_Pivot_R3S3_Breakout_1dTrend_Volume_Momentum
# Hypothesis: Combines 1d Camarilla R3/S3 breakout with 1d EMA34 trend and 4h RSI momentum filter.
# Trades only when breakout aligns with higher timeframe trend and momentum confirms direction.
# Designed for low turnover (target 20-30 trades/year) to minimize fee drag in 2025 ranging markets.
# Uses momentum to avoid false breakouts and improve win rate in both bull and bear regimes.

name = "4h_Camarilla_Pivot_R3S3_Breakout_1dTrend_Volume_Momentum"
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
    volume = prices['volume'].values
    
    # === 1d Camarilla Pivot Levels ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r3 = pivot + (range_1d * 1.1 / 2)
    s3 = pivot - (range_1d * 1.1 / 2)
    
    # Align 1d Camarilla levels to 4h
    r3_4h = align_htf_to_ltf(prices, df_1d, r3)
    s3_4h = align_htf_to_ltf(prices, df_1d, s3)
    
    # === 1d EMA34 Trend Filter ===
    ema34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema34_1d_4h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === 4h RSI(14) Momentum Filter ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # === Volume Spike Filter (20-period EMA) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 1.5  # Require 1.5x average volume
    
    # === Signal Parameters ===
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 60  # covers EMA34 and RSI
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_4h[i]) or np.isnan(s3_4h[i]) or 
            np.isnan(ema34_1d_4h[i]) or np.isnan(rsi[i]) or 
            np.isnan(volume_ok[i])):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Break above R3 + above 1d EMA34 + RSI > 50 + volume spike
            if (close[i] > r3_4h[i] and 
                close[i] > ema34_1d_4h[i] and 
                rsi[i] > 50 and 
                volume_ok[i]):
                signals[i] = position_size
                position = 1
            # Short: Break below S3 + below 1d EMA34 + RSI < 50 + volume spike
            elif (close[i] < s3_4h[i] and 
                  close[i] < ema34_1d_4h[i] and 
                  rsi[i] < 50 and 
                  volume_ok[i]):
                signals[i] = -position_size
                position = -1
        else:
            # Exit conditions: close below/above opposite level
            if position == 1:
                # Exit: Price closes below S3 (opposite level)
                if close[i] < s3_4h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Exit: Price closes above R3 (opposite level)
                if close[i] > r3_4h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals