#!/usr/bin/env python3
# 12h_Camarilla_R3S3_Breakout_1dTrend_Volume
# Hypothesis: Camarilla pivot levels (R3/S3) from 1d act as strong support/resistance. 
# Breakout above R3 or below S3 with 1d trend filter (price above/below 20 EMA) and volume confirmation captures momentum.
# Works in bull/bear by following higher timeframe trend. Targets 15-30 trades/year on 12h to avoid fee drag.

name = "12h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "12h"
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
    
    # === 1d OHLC for Camarilla calculation ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC (Camarilla uses prior day)
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = np.nan  # First day has no previous
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Camarilla levels: R3, S3
    # R3 = prev_close + 1.1 * (prev_high - prev_low) / 2
    # S3 = prev_close - 1.1 * (prev_high - prev_low) / 2
    camarilla_r3 = prev_close + 1.1 * (prev_high - prev_low) / 2
    camarilla_s3 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # Align to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # === 1d EMA20 for trend filter ===
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # === Volume confirmation (20-period average on 12h) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_20_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter: above average
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # LONG: Price breaks above R3, above 1d EMA20, volume confirmation
            if close[i] > camarilla_r3_aligned[i] and close[i] > ema_20_aligned[i] and vol_ok:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3, below 1d EMA20, volume confirmation
            elif close[i] < camarilla_s3_aligned[i] and close[i] < ema_20_aligned[i] and vol_ok:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price falls below S3 or below EMA20
            if close[i] < camarilla_s3_aligned[i] or close[i] < ema_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises above R3 or above EMA20
            if close[i] > camarilla_r3_aligned[i] or close[i] > ema_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals