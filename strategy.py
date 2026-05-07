#!/usr/bin/env python3
name = "4h_Camarilla_R3_S3_Breakout_12hTrend_VolumeSpike"
timeframe = "4h"
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
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # 12h volume spike: > 1.8x 24-period average (12 days)
    vol_ma_12h = pd.Series(df_12h['volume']).rolling(window=24, min_periods=24).mean().values
    vol_spike_12h = df_12h['volume'].values > 1.8 * vol_ma_12h
    vol_spike_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_spike_12h)
    
    # 4h Camarilla levels (R3, S3) from previous day
    # Calculate from daily data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's high, low, close
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla R3 and S3
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 24)  # Wait for EMA50 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(vol_spike_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close above R3, 12h trend up, volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema50_12h_aligned[i] and 
                vol_spike_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Close below S3, 12h trend down, volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema50_12h_aligned[i] and 
                  vol_spike_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close below S3 or trend turns down
            if (close[i] < camarilla_s3_aligned[i] or 
                close[i] < ema50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above R3 or trend turns up
            if (close[i] > camarilla_r3_aligned[i] or 
                close[i] > ema50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike confirmation.
# Long when price breaks above R3 (strong resistance), 12h trend is up (price > EMA50), 
# and 12h volume spike confirms institutional participation.
# Short when price breaks below S3 (strong support), 12h trend is down, and volume spike present.
# Uses 12h timeframe for trend/volume to filter 4h noise, 4h for entry timing.
# Camarilla levels provide mathematically derived support/resistance with institutional relevance.
# Volume spike (>1.8x average) ensures conviction, reducing false breakouts.
# Discrete 0.25 position size limits risk. Target: 25-40 trades/year to minimize fee drag.
# Works in bull markets (breakouts with trend) and bear markets (breakdowns with trend).