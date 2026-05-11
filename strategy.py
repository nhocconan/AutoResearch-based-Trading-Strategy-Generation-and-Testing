#!/usr/bin/env python3
# 4h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike
# Hypothesis: Camarilla R3/S3 breakouts filtered by 1-day EMA34 trend and volume spike.
# Long when: price breaks above R3, 1d EMA34 rising, volume > 2x 20-period average.
# Short when: price breaks below S3, 1d EMA34 falling, volume > 2x 20-period average.
# Exit when price crosses back to Camarilla pivot point or 1d EMA34 trend reverses.
# Camarilla levels provide institutional support/resistance, EMA34 filters counter-trend moves,
# volume confirms breakout strength. Works in bull markets via breakouts, in bear via breakdowns.

name = "4h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for EMA34 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Camarilla levels from previous day ---
    # Using previous day's OHLC to avoid look-ahead
    camarilla_pivot = np.full(n, np.nan)
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    
    for i in range(1, n):
        # Previous day's OHLC
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        
        # Camarilla calculations
        pivot = (prev_high + prev_low + prev_close) / 3
        range_val = prev_high - prev_low
        
        camarilla_pivot[i] = pivot
        camarilla_r3[i] = pivot + (range_val * 1.1 / 4)  # R3
        camarilla_s3[i] = pivot - (range_val * 1.1 / 4)  # S3
    
    # --- 1d EMA34 trend ---
    close_1d = df_1d['close'].values
    ema_1d = np.full(len(close_1d), np.nan)
    for i in range(34, len(close_1d)):
        if i == 34:
            ema_1d[i] = np.mean(close_1d[0:34])
        else:
            ema_1d[i] = (close_1d[i] * 2 / (34 + 1)) + (ema_1d[i-1] * (33 / (34 + 1)))
    
    # EMA slope (rising/falling)
    ema_slope = np.full(len(close_1d), np.nan)
    for i in range(35, len(close_1d)):
        ema_slope[i] = ema_1d[i] - ema_1d[i-1]
    
    # Align 1d EMA and slope to 4h
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    ema_slope_aligned = align_htf_to_ltf(prices, df_1d, ema_slope)
    
    # --- Volume confirmation (volume > 20-period average) ---
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for Camarilla (1), EMA34 (34), volume MA (20)
    start_idx = max(1, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(camarilla_pivot[i]) or
            np.isnan(camarilla_r3[i]) or
            np.isnan(camarilla_s3[i]) or
            np.isnan(ema_1d_aligned[i]) or
            np.isnan(ema_slope_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Camarilla breakout conditions
        breakout_up = close[i] > camarilla_r3[i]
        breakout_down = close[i] < camarilla_s3[i]
        
        # Volume spike condition
        vol_spike = volume[i] > vol_ma[i] * 2.0  # 100% above average
        
        if position == 0:
            if breakout_up and ema_slope_aligned[i] > 0 and vol_spike:
                # Long: upward breakout + rising EMA34 + volume spike
                signals[i] = 0.25
                position = 1
            elif breakout_down and ema_slope_aligned[i] < 0 and vol_spike:
                # Short: downward breakout + falling EMA34 + volume spike
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price falls to pivot OR EMA34 slope turns negative
                if close[i] < camarilla_pivot[i] or ema_slope_aligned[i] < 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price rises to pivot OR EMA34 slope turns positive
                if close[i] > camarilla_pivot[i] or ema_slope_aligned[i] > 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals