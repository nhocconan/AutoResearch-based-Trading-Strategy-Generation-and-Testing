#!/usr/bin/env python3
# 12h_Camarilla_R3_S3_Breakout_1wTrend_Volume
# Hypothesis: 12h Camarilla R3/S3 breakout filtered by 1-week EMA trend and volume spike.
# Long when: price breaks above R3, 1w EMA50 rising, volume > 1.5x 20-period average.
# Short when: price breaks below S3, 1w EMA50 falling, volume > 1.5x 20-period average.
# Exit when price crosses back to H5/L5 or 1w EMA50 trend reverses.
# Camarilla levels provide institutional support/resistance, weekly trend filters counter-trend moves,
# volume confirms breakout strength. Designed for fewer trades (~20-50/year) to minimize fee drag.

name = "12h_Camarilla_R3_S3_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Get 1w data for EMA50 trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1w EMA50 trend ---
    close_1w = df_1w['close'].values
    ema_1w = np.full(len(close_1w), np.nan)
    for i in range(50, len(close_1w)):
        if i == 50:
            ema_1w[i] = np.mean(close_1w[0:50])
        else:
            ema_1w[i] = (close_1w[i] * 2 / (50 + 1)) + (ema_1w[i-1] * (49 / (50 + 1)))
    
    # EMA slope (rising/falling)
    ema_slope_1w = np.full(len(close_1w), np.nan)
    for i in range(51, len(close_1w)):
        ema_slope_1w[i] = ema_1w[i] - ema_1w[i-1]
    
    # Align 1w EMA and slope to 12h
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    ema_slope_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_slope_1w)
    
    # --- Camarilla levels from 1d (H5, L5, H3, L3) ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_H5 = np.full(len(close_1d), np.nan)
    camarilla_L5 = np.full(len(close_1d), np.nan)
    camarilla_H3 = np.full(len(close_1d), np.nan)
    camarilla_L3 = np.full(len(close_1d), np.nan)
    
    for i in range(len(close_1d)):
        if i == 0:
            camarilla_H5[i] = np.nan
            camarilla_L5[i] = np.nan
            camarilla_H3[i] = np.nan
            camarilla_L3[i] = np.nan
        else:
            # Previous day's range
            prev_high = high_1d[i-1]
            prev_low = low_1d[i-1]
            prev_close = close_1d[i-1]
            range_ = prev_high - prev_low
            
            camarilla_H5[i] = prev_close + range_ * 1.1 / 2
            camarilla_L5[i] = prev_close - range_ * 1.1 / 2
            camarilla_H3[i] = prev_close + range_ * 1.1 / 4
            camarilla_L3[i] = prev_close - range_ * 1.1 / 4
    
    # Align Camarilla levels to 12h
    camarilla_H5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H5)
    camarilla_L5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L5)
    camarilla_H3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H3)
    camarilla_L3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L3)
    
    # --- Volume confirmation (volume > 20-period average) ---
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for 1w EMA50, Camarilla (need prev day), and volume MA(20)
    start_idx = max(50, 1, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_1w_aligned[i]) or
            np.isnan(ema_slope_1w_aligned[i]) or
            np.isnan(camarilla_H3_aligned[i]) or
            np.isnan(camarilla_L3_aligned[i]) or
            np.isnan(camarilla_H5_aligned[i]) or
            np.isnan(camarilla_L5_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Camarilla breakout conditions
        breakout_up = close[i] > camarilla_H3_aligned[i]
        breakout_down = close[i] < camarilla_L3_aligned[i]
        
        # Volume spike condition
        vol_spike = volume[i] > vol_ma[i] * 1.5  # 50% above average
        
        if position == 0:
            if breakout_up and ema_slope_1w_aligned[i] > 0 and vol_spike:
                # Long: upward breakout above H3 + rising 1w EMA50 + volume spike
                signals[i] = 0.25
                position = 1
            elif breakout_down and ema_slope_1w_aligned[i] < 0 and vol_spike:
                # Short: downward breakout below L3 + falling 1w EMA50 + volume spike
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price falls back to L3 OR 1w EMA50 slope turns negative
                if close[i] < camarilla_L3_aligned[i] or ema_slope_1w_aligned[i] < 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price rises back to H3 OR 1w EMA50 slope turns positive
                if close[i] > camarilla_H3_aligned[i] or ema_slope_1w_aligned[i] > 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals