#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    """
    12h Camarilla R1/S1 breakout with 1w trend filter and volume confirmation.
    Long: Close breaks above R1 AND close > 1w EMA(34) AND volume > 1.5x 12h avg volume
    Short: Close breaks below S1 AND close < 1w EMA(34) AND volume > 1.5x 12h avg volume
    Exit: Opposite signal or price reverts to Camarilla pivot
    Uses 1d data for Camarilla levels (HLC from previous day)
    Target: 15-35 trades/year on 12h timeframe
    """
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation (previous day's HLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    # Calculate 1w EMA(34) for trend filter
    close_1w = pd.Series(df_1w['close'].values)
    ema34_1w = close_1w.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate average volume for volume confirmation (20-period SMA)
    volume_s = pd.Series(volume)
    vol_ma20 = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if trend or volume data not ready
        if np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get previous day's HLC for Camarilla calculation
        prev_idx_1d = i // 16  # Approximate: 16x 12h bars per day
        if prev_idx_1d < 1 or prev_idx_1d >= len(df_1d):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Previous day's high, low, close
        prev_high = df_1d['high'].iloc[prev_idx_1d - 1]
        prev_low = df_1d['low'].iloc[prev_idx_1d - 1]
        prev_close = df_1d['close'].iloc[prev_idx_1d - 1]
        
        # Calculate Camarilla levels
        range_val = prev_high - prev_low
        if range_val <= 0:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Camarilla R1 and S1 levels
        r1 = prev_close + (range_val * 1.1 / 12)
        s1 = prev_close - (range_val * 1.1 / 12)
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > (1.5 * vol_ma20[i])
        
        if position == 0:
            # Long: Break above R1 with trend and volume confirmation
            if close[i] > r1 and close[i] > ema34_1w_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Break below S1 with trend and volume confirmation
            elif close[i] < s1 and close[i] < ema34_1w_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price breaks below S1 or trend fails
            if close[i] < s1 or close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price breaks above R1 or trend fails
            if close[i] > r1 or close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals