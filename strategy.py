#!/usr/bin/env python3
# 4h_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: Camarilla pivot R3/S3 breakouts on 4h with 1d EMA34 trend filter and volume confirmation.
# Works in bull (breakouts continue with trend) and bear (mean-reversion at extremes via trend filter).
# Target: 20-50 trades/year (80-200 total over 4 years).

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
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
    
    # === 1d Trend Filter (EMA34) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === Camarilla Pivot Levels (from previous day) ===
    # Calculate pivot points using previous day's OHLC
    # R4 = C + ((H-L) * 1.5)
    # R3 = C + ((H-L) * 1.25)
    # R2 = C + ((H-L) * 1.166)
    # R1 = C + ((H-L) * 1.083)
    # PP = (H + L + C) / 3
    # S1 = C - ((H-L) * 1.083)
    # S2 = C - ((H-L) * 1.166)
    # S3 = C - ((H-L) * 1.25)
    # S4 = C - ((H-L) * 1.5)
    
    # We need previous day's OHLC for current day's levels
    # Since we're on 4h timeframe, we'll calculate daily pivots and align
    
    # Get daily OHLC for pivot calculation
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC (shifted by 1 to avoid look-ahead)
    prev_high = df_daily['high'].shift(1).values
    prev_low = df_daily['low'].shift(1).values
    prev_close = df_daily['close'].shift(1).values
    
    # Calculate Camarilla levels
    # R3 = C + ((H-L) * 1.25)
    # S3 = C - ((H-L) * 1.25)
    camarilla_r3 = prev_close + ((prev_high - prev_low) * 1.25)
    camarilla_s3 = prev_close - ((prev_high - prev_low) * 1.25)
    
    # Align to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_daily, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_daily, camarilla_s3)
    
    # === Volume Confirmation (20-period average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend direction
        trend_up = close[i] > ema_34_1d_aligned[i]
        trend_down = close[i] < ema_34_1d_aligned[i]
        
        # Volume filter: above average
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # LONG: Price breaks above R3 with volume and uptrend
            if (close[i] > camarilla_r3_aligned[i] and vol_ok and trend_up):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 with volume and downtrend
            elif (close[i] < camarilla_s3_aligned[i] and vol_ok and trend_down):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price returns to S3 level or trend changes
            if (close[i] < camarilla_s3_aligned[i] or not trend_up):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to R3 level or trend changes
            if (close[i] > camarilla_r3_aligned[i] or not trend_down):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals