#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_1dTrend_VolumeS
Hypothesis: Use 1d trend (EMA34) and volume confirmation to filter breakouts at Camarilla R3/S3 levels. Designed for ~25 trades/year per symbol to avoid fee drag while capturing high-probability trend continuation moves.
"""

name = "4h_Camarilla_R3S3_Breakout_1dTrend_VolumeS"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for trend and volume context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # 4h OHLCV
    close_4h = prices['close'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    volume_4h = prices['volume'].values
    
    # --- 1d EMA34 for trend direction ---
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # --- 1d Volume average for confirmation ---
    vol_avg_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_avg_4h = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # --- 4h Volume spike detector ---
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # --- 1d Previous day OHLC for Camarilla (R3/S3) ---
    prev_high = np.roll(df_1d['high'].values, 1)
    prev_low = np.roll(df_1d['low'].values, 1)
    prev_close = np.roll(df_1d['close'].values, 1)
    prev_high[0] = df_1d['high'].values[0]
    prev_low[0] = df_1d['low'].values[0]
    prev_close[0] = df_1d['close'].values[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    
    r3 = pivot + (range_val * 1.1 / 2.0)
    s3 = pivot - (range_val * 1.1 / 2.0)
    
    r3_4h = align_htf_to_ltf(prices, df_1d, r3)
    s3_4h = align_htf_to_ltf(prices, df_1d, s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 35  # warmup
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema34_4h[i]) or np.isnan(vol_avg_4h[i]) or 
            np.isnan(r3_4h[i]) or np.isnan(s3_4h[i])):
            if position != 0:
                # Exit if conditions deteriorate
                if position == 1 and (close_4h[i] < ema34_4h[i] or volume_4h[i] < vol_avg_4h[i]):
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and (close_4h[i] > ema34_4h[i] or volume_4h[i] < vol_avg_4h[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Volume confirmation: current 4h volume > 1.5x 1d average volume
        vol_spike = volume_4h[i] > 1.5 * vol_avg_4h[i]
        
        if position == 0:
            # Look for breakout entries with volume confirmation
            if close_4h[i] > r3_4h[i] and close_4h[i] > ema34_4h[i] and vol_spike:
                signals[i] = 0.25  # long breakout
                position = 1
                entry_price = close_4h[i]
            elif close_4h[i] < s3_4h[i] and close_4h[i] < ema34_4h[i] and vol_spike:
                signals[i] = -0.25  # short breakdown
                position = -1
                entry_price = close_4h[i]
        else:
            # Manage existing position
            if position == 1:
                # Long: exit if price breaks below EMA34 or volume drops
                if close_4h[i] < ema34_4h[i] or volume_4h[i] < vol_avg_4h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short: exit if price breaks above EMA34 or volume drops
                if close_4h[i] > ema34_4h[i] or volume_4h[i] < vol_avg_4h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals