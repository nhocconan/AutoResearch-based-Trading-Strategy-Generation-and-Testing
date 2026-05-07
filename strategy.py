#!/usr/bin/env python3
name = "12h_Camarilla_R3S3_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

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
    
    # Load 1w and 1d data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 2 or len(df_1d) < 2:
        return np.zeros(n)
    
    # 1w trend: 50 EMA
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # 1d Camarilla pivot levels (using previous day's OHLC)
    # R3 = close + 1.1 * (high - low)
    # S3 = close - 1.1 * (high - low)
    # R4 = close + 1.5 * (high - low)
    # S4 = close - 1.5 * (high - low)
    # We'll use R3/S3 for entry and R4/S4 for stop (implemented via signal=0)
    # But since we need previous day's data, we shift by 1
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for previous day
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d)
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d)
    camarilla_r4 = close_1d + 1.5 * (high_1d - low_1d)
    camarilla_s4 = close_1d - 1.5 * (high_1d - low_1d)
    
    # Align to 12h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # 12h volume spike: > 1.5x 24-period average (12 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_spike = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(24, 50)  # Wait for volume MA and 1w EMA50
    
    for i in range(start_idx, n):
        if np.isnan(ema50_1w_aligned[i]) or np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R3, above 1w EMA50, volume spike
            if close[i] > r3_1d_aligned[i] and close[i] > ema50_1w_aligned[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3, below 1w EMA50, volume spike
            elif close[i] < s3_1d_aligned[i] and close[i] < ema50_1w_aligned[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price breaks below S4 or below 1w EMA50
            if close[i] < s4_1d_aligned[i] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price breaks above R4 or above 1w EMA50
            if close[i] > r4_1d_aligned[i] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 12h Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume confirmation.
# Long when price breaks above R3 (Camarilla resistance level 3), above 1w EMA50 (uptrend), and volume spike confirms.
# Short when price breaks below S3 (Camarilla support level 3), below 1w EMA50 (downtrend), and volume spike confirms.
# Uses 1w timeframe for trend to avoid whipsaws, 12h for entry timing.
# Volume spike (>1.5x average) ensures conviction. Uses Camarilla R4/S4 as stop levels.
# Discrete 0.25 position size limits risk. Works in both bull and bear markets by following 1w trend.
# Target: 12-37 trades/year to minimize fee drag while capturing significant moves.