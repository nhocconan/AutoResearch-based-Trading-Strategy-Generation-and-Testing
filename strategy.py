#!/usr/bin/env python3
name = "12h_Camarilla_R2_S2_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA(50) for trend filter
    ema_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Load 1d data ONCE before loop for Pivot Points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Pivot Points from previous 1d for R2/S2
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3
    r2 = pivot + (prev_high - prev_low)  # R2 = P + (H - L)
    s2 = pivot - (prev_high - prev_low)  # S2 = P - (H - L)
    
    # Align Pivot levels to 12h
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # Volume filter: > 1.8x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_filter = volume > 1.8 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for EMA
    
    for i in range(start_idx, n):
        if np.isnan(ema_1w_aligned[i]) or np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above R2 with 1w uptrend and volume
            if (close[i] > r2_aligned[i] and close[i] > ema_1w_aligned[i] and vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Break below S2 with 1w downtrend and volume
            elif (close[i] < s2_aligned[i] and close[i] < ema_1w_aligned[i] and vol_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close below S2 or trend change
            if close[i] < s2_aligned[i] or close[i] < ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above R2 or trend change
            if close[i] > r2_aligned[i] or close[i] > ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 12h Camarilla R2/S2 breakout with 1w EMA(50) trend filter and volume confirmation.
# R2/S2 are stronger support/resistance levels than R1/S1, offering cleaner breakouts.
# 1w EMA(50) ensures alignment with weekly trend, reducing whipsaw and capturing major moves.
# Volume confirms institutional participation. Position size 0.25 limits drawdown.
# Target: ~15-25 trades/year to avoid fee drag while capturing significant trends.