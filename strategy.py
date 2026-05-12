#!/usr/bin/env python3
# 4h_Camarilla_R3S3_Breakout_1dTrend_Volume
# Hypothesis: Camarilla R3/S3 breakouts with daily trend and volume filter capture momentum with controlled risk.
# Uses 1d EMA50 for trend filter and volume > 1.5x 20-period average for confirmation.
# Designed for low trade frequency and high win rate in both bull and bear markets.
# Uses discrete position sizing (0.25) to limit drawdown and minimize trade frequency.

name = "4h_Camarilla_R3S3_Breakout_1dTrend_Volume"
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
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Previous day's Camarilla levels (R3, S3)
    # Camarilla: R3 = close + (high - low) * 1.1/4, S3 = close - (high - low) * 1.1/4
    # Use previous day's close, high, low
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(df_1d['high'].values, 1)
    prev_low_1d = np.roll(df_1d['low'].values, 1)
    prev_close_1d[0] = np.nan
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    
    camarilla_range = (prev_high_1d - prev_low_1d) * 1.1 / 4
    r3 = prev_close_1d + camarilla_range
    s3 = prev_close_1d - camarilla_range
    
    # Align Camarilla levels to 4h
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Get aligned 1d close for trend filter
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
        close_1d_current = close_1d_aligned[i]
        
        trend_up = close_1d_current > ema50_1d_aligned[i]
        trend_down = close_1d_current < ema50_1d_aligned[i]
        
        vol_confirm = volume_confirm[i]
        
        if position == 0:
            # LONG: Close > R3 AND 1d uptrend AND volume confirmation
            if close[i] > r3_aligned[i] and trend_up and vol_confirm:
                signals[i] = 0.25
                position = 1
            # SHORT: Close < S3 AND 1d downtrend AND volume confirmation
            elif close[i] < s3_aligned[i] and trend_down and vol_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Close < S3 (reversal to opposite level) OR trend weakens
            if close[i] < s3_aligned[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close > R3 (reversal to opposite level) OR trend weakens
            if close[i] > r3_aligned[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals