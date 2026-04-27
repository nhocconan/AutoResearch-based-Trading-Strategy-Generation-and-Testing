#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeS_v3
Hypothesis: Tighten entry by requiring volume spike >2.5x average and adding a minimum holding period of 4 bars to reduce whipsaw. Long when price > EMA50 and breaks above R1, short when price < EMA50 and breaks below S1. Exit on opposite level touch. Uses 1d Camarilla levels and 12h EMA50 trend filter. Designed for 20-40 trades/year to avoid fee drag while maintaining edge in bull/bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels from 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla R1 and S1 levels
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 6
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 6
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    ema_50 = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Volume confirmation: current volume > 2.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for volume average and EMA
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        bars_since_entry += 1
        
        # Skip if any data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        camarilla_r1_val = camarilla_r1_aligned[i]
        camarilla_s1_val = camarilla_s1_aligned[i]
        ema_50_val = ema_50_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Require minimum 4 bars since last exit to avoid churn
            if bars_since_entry >= 4:
                # Long: price breaks above R1 with volume confirmation AND above 12h EMA50 (uptrend)
                if close[i] > camarilla_r1_val and vol_conf and close[i] > ema_50_val:
                    signals[i] = size
                    position = 1
                    bars_since_entry = 0
                # Short: price breaks below S1 with volume confirmation AND below 12h EMA50 (downtrend)
                elif close[i] < camarilla_s1_val and vol_conf and close[i] < ema_50_val:
                    signals[i] = -size
                    position = -1
                    bars_since_entry = 0
        elif position == 1:
            # Exit long: price breaks below S1 (opposite level)
            if close[i] < camarilla_s1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price breaks above R1 (opposite level)
            if close[i] > camarilla_r1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeS"
timeframe = "4h"
leverage = 1.0