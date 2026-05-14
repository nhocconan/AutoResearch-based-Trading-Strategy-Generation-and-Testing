#!/usr/bin/env python3
# Hypothesis: 6h strategy using weekly pivot points (Camarilla) with daily trend filter and volume confirmation.
# Long when price breaks above weekly R3 with price > 1d EMA50 (bullish trend) and 6h volume > 1.8x 24-period average.
# Short when price breaks below weekly S3 with price < 1d EMA50 (bearish trend) and 6h volume > 1.8x 24-period average.
# Exit on opposite weekly Camarilla level (S3 for longs, R3 for shorts).
# Uses weekly Camarilla for structural levels (more significant than daily), 1d EMA50 for trend filter,
# and volume spike to confirm participation. Target: 60-120 total trades over 4 years (15-30/year).

name = "6h_Camarilla_Weekly_R3S3_Breakout_1dEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 6h Indicators (LTF) ---
    # 6h volume confirmation: > 1.8x 24-period average (4 days)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike_6h = volume > (1.8 * vol_ma_24)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) - trend filter
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # --- Weekly Camarilla Pivot Points (Prior Week OHLC) ---
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    range_val = weekly_high - weekly_low
    camarilla_r3_weekly = weekly_close + (range_val * 1.1 / 2)  # R3
    camarilla_s3_weekly = weekly_close - (range_val * 1.1 / 2)  # S3
    
    # Align weekly levels to 6h timeframe (wait for weekly close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3_weekly)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3_weekly)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(ema_50_aligned[i]) or
            np.isnan(volume_spike_6h[i]) or
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above weekly R3 + price > 1d EMA50 (bullish) + 6h volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_50_aligned[i] and 
                volume_spike_6h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below weekly S3 + price < 1d EMA50 (bearish) + 6h volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_50_aligned[i] and 
                  volume_spike_6h[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below weekly S3
            if close[i] < camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above weekly R3
            if close[i] > camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals