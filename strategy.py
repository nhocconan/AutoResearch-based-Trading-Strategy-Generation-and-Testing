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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d Camarilla R3 and S3 levels
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    pivot = (prev_high + prev_low + prev_close) / 3
    range_ = prev_high - prev_low
    r3 = pivot + (range_ * 1.1 / 2)
    s3 = pivot - (range_ * 1.1 / 2)
    
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # 1w EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # 12h volume spike: > 1.5x 24-period average (12 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_spike = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(24, 34)  # Wait for volume MA and weekly EMA
    
    for i in range(start_idx, n):
        if np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(ema34_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close above R3, above weekly EMA34, volume spike
            if close[i] > r3_aligned[i] and close[i] > ema34_1w_aligned[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close below S3, below weekly EMA34, volume spike
            elif close[i] < s3_aligned[i] and close[i] < ema34_1w_aligned[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close below S3 or below weekly EMA34
            if close[i] < s3_aligned[i] or close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above R3 or above weekly EMA34
            if close[i] > r3_aligned[i] or close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 12h Camarilla R3/S3 breakout with weekly trend filter and volume confirmation.
# Long when price breaks above R3 (strong resistance), above weekly EMA34 (bullish trend),
# and volume spike confirms conviction. Short when price breaks below S3 (strong support),
# below weekly EMA34 (bearish trend), with volume spike.
# Uses weekly timeframe for trend to avoid whipsaws, 12h for entry timing.
# Camarilla R3/S3 are strong institutional levels; breakouts often lead to sustained moves.
# Volume spike (>1.5x average) ensures institutional participation.
# Discrete 0.25 position size limits risk and reduces fee churn.
# Works in bull markets (breakouts above R3) and bear markets (breakdowns below S3).
# Target: 15-35 trades/year to minimize fee drag while capturing sustained moves.