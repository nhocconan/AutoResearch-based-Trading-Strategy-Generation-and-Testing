#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_Volume_Trend
Hypothesis: Use daily Camarilla R3/S3 levels as support/resistance, breakout with volume confirmation, and trend filter from 1-hour EMA50 to align with intraday momentum. Designed for low trade frequency (<25/year) to minimize fee drag on 4h timeframe. Works in bull/bear by filtering breakouts in trend direction only.
"""

name = "4h_Camarilla_R3_S3_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Get Daily OHLC for Camarilla Calculation ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Prior day's OHLC (shifted by 1 to avoid look-ahead)
    prior_high = df_1d['high'].shift(1).values
    prior_low = df_1d['low'].shift(1).values
    prior_close = df_1d['close'].shift(1).values
    
    # Align to 4h timeframe
    prior_high_4h = align_htf_to_ltf(prices, df_1d, prior_high)
    prior_low_4h = align_htf_to_ltf(prices, df_1d, prior_low)
    prior_close_4h = align_htf_to_ltf(prices, df_1d, prior_close)
    
    # Camarilla R3 and S3 levels
    R3 = prior_close_4h + (prior_high_4h - prior_low_4h) * 1.1 / 4
    S3 = prior_close_4h - (prior_high_4h - prior_low_4h) * 1.1 / 4
    
    # === 1-hour EMA50 Trend Filter ===
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 2:
        return np.zeros(n)
    ema50_1h = pd.Series(df_1h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1h_4h = align_htf_to_ltf(prices, df_1h, ema50_1h)
    
    # === Volume Spike Filter (2x 20-period EMA) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_ok = volume > vol_ema20 * 2.0
    
    # === Signal Parameters ===
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers EMA and Camarilla calculation)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(R3[i]) or np.isnan(S3[i]) or 
            np.isnan(ema50_1h_4h[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Close crosses above R3 with uptrend (close > EMA50) and volume spike
            if (close[i] > R3[i] and 
                close[i] > ema50_1h_4h[i] and volume_ok[i]):
                signals[i] = position_size
                position = 1
            # Short: Close crosses below S3 with downtrend (close < EMA50) and volume spike
            elif (close[i] < S3[i] and 
                  close[i] < ema50_1h_4h[i] and volume_ok[i]):
                signals[i] = -position_size
                position = -1
        else:
            # Exit: Close crosses back through the opposite Camarilla level
            if position == 1:
                if close[i] < S3[i]:  # Exit long if price breaks below S3
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                if close[i] > R3[i]:  # Exit short if price breaks above R3
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals