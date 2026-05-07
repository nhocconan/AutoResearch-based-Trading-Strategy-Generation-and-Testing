#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Camarilla Pivot R3/S3 breakout with 12-hour EMA50 trend filter and volume spike confirmation.
# Long when: Close breaks above Camarilla R3 AND 12h EMA50 rising AND volume > 2.0 * 20-period EMA(volume).
# Short when: Close breaks below Camarilla S3 AND 12h EMA50 falling AND volume > 2.0 * 20-period EMA(volume).
# Uses Camarilla levels for institutional support/resistance, EMA50 for trend direction, volume for conviction.
# Designed for low trade frequency (target: 25-35/year) to minimize fee drag and improve generalization.
# Works in bull markets via upward breaks of R3 and in bear markets via downward breaks of S3.
name = "4h_Camarilla_R3S3_12hEMA50_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels (R3, S3) from previous day
    # Using prior day's high, low, close
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    
    # First day: use first available values
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    r3 = pivot + (range_hl * 1.1 / 4.0)  # R3 = PP + 1.1 * (H-L) / 4
    s3 = pivot - (range_hl * 1.1 / 4.0)  # S3 = PP - 1.1 * (H-L) / 4
    
    # Load 12h data for EMA50 trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # EMA50 slope for trend direction (rising/falling)
    ema_50_prev = np.roll(ema_50_12h_aligned, 1)
    ema_50_prev[0] = ema_50_12h_aligned[0]
    ema_50_rising = ema_50_12h_aligned > ema_50_prev
    ema_50_falling = ema_50_12h_aligned < ema_50_prev
    
    # Volume confirmation: current volume > 2.0 * 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for EMA50
    
    for i in range(start_idx, n):
        if (np.isnan(r3[i]) or np.isnan(s3[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close > R3 AND EMA50 rising AND volume spike
            long_condition = (close[i] > r3[i]) and ema_50_rising[i] and volume_spike[i]
            # Short: Close < S3 AND EMA50 falling AND volume spike
            short_condition = (close[i] < s3[i]) and ema_50_falling[i] and volume_spike[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close < EMA50 or EMA50 turns flat/falling
            if close[i] < ema_50_12h_aligned[i] or not ema_50_rising[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close > EMA50 or EMA50 turns flat/rising
            if close[i] > ema_50_12h_aligned[i] or not ema_50_falling[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals