#!/usr/bin/env python3
name = "1d_1w_Camarilla_R3S3_Breakout_WeeklyTrend_Volume"
timeframe = "1d"
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
    
    # Load weekly data ONCE before loop for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate Weekly Pivot Points from previous week
    prev_high = df_1w['high'].shift(1).values
    prev_low = df_1w['low'].shift(1).values
    prev_close = df_1w['close'].shift(1).values
    
    # Classic floor trader pivot: P = (H+L+C)/3
    pivot = (prev_high + prev_low + prev_close) / 3
    # Weekly R3 and S3 levels
    r3 = pivot + 2 * (prev_high - prev_low)  # R3 = P + 2*(H-L)
    s3 = pivot - 2 * (prev_high - prev_low)  # S3 = P - 2*(H-L)
    
    # Align Weekly levels to 1d
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Load weekly data for trend filter (EMA21 on weekly close)
    ema_1w = pd.Series(df_1w['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume filter: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 21  # Wait for weekly EMA
    
    for i in range(start_idx, n):
        if np.isnan(ema_1w_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above R3 with weekly uptrend and volume
            if (close[i] > r3_aligned[i] and close[i] > ema_1w_aligned[i] and vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Break below S3 with weekly downtrend and volume
            elif (close[i] < s3_aligned[i] and close[i] < ema_1w_aligned[i] and vol_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close below S3 or trend change
            if close[i] < s3_aligned[i] or close[i] < ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above R3 or trend change
            if close[i] > r3_aligned[i] or close[i] > ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 1d Weekly Pivot R3/S3 breakout with 1w EMA(21) trend filter and volume confirmation.
# Weekly R3/S3 are strong institutional levels that act as magnet/resistance.
# Breakouts beyond these levels indicate strong momentum with follow-through.
# 1w EMA(21) ensures alignment with weekly trend, reducing whipsaw in choppy markets.
# Volume confirms institutional participation. Position size 0.25 limits drawdown.
# Target: ~10-20 trades/year to avoid fee dust while capturing significant weekly moves.