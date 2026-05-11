# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# 4h_1d_Camarilla_R1S1_Breakout_VolumeTrend
# Hypothesis: Price tends to respect Camarilla pivot levels (R1/S1) derived from prior day's range.
# Enter long when price breaks above S1 with bullish 1d EMA trend and volume confirmation.
# Enter short when price breaks below R1 with bearish 1d EMA trend and volume confirmation.
# Exit when price reverses to the opposite Camarilla level or trend weakens.
# Uses 1d Camarilla levels for structure, 1d EMA34 for trend filter, 4h volume spike for confirmation.
# Designed for 4h timeframe to target 20-50 trades/year, avoiding excessive turnover.

name = "4h_1d_Camarilla_R1S1_Breakout_VolumeTrend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for Camarilla calculation and EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d Camarilla levels (based on prior day's range) ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_r1 = np.full(len(close_1d), np.nan)
    camarilla_s1 = np.full(len(close_1d), np.nan)
    
    for i in range(len(close_1d)):
        if i == 0:
            # First day: use same day's range (no prior)
            rng = high_1d[i] - low_1d[i]
            camarilla_r1[i] = close_1d[i] + 1.1 * rng / 12
            camarilla_s1[i] = close_1d[i] - 1.1 * rng / 12
        else:
            # Use prior day's range
            rng = high_1d[i-1] - low_1d[i-1]
            camarilla_r1[i] = close_1d[i-1] + 1.1 * rng / 12
            camarilla_s1[i] = close_1d[i-1] - 1.1 * rng / 12
    
    # --- 1d EMA34 trend ---
    ema_34 = np.full(len(close_1d), np.nan)
    for i in range(len(close_1d)):
        if i < 34:
            ema_34[i] = np.nan
        elif i == 34:
            ema_34[i] = np.mean(close_1d[0:34])
        else:
            ema_34[i] = (close_1d[i] * 2 / (34 + 1)) + (ema_34[i-1] * (33 / (34 + 1)))
    
    # EMA slope for trend direction
    ema_slope = np.full(len(close_1d), np.nan)
    for i in range(35, len(close_1d)):
        ema_slope[i] = ema_34[i] - ema_34[i-1]
    
    # --- 4h ATR(14) for volatility-based sizing (optional, not used in entry) ---
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            atr[i] = np.mean(tr[0:14])
        else:
            atr[i] = (tr[i] * 1 / 14) + (atr[i-1] * 13 / 14)
    
    # --- 4h volume MA(20) for confirmation ---
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align 1d indicators to 4h
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    ema_slope_aligned = align_htf_to_ltf(prices, df_1d, ema_slope)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max(1d needs 35 bars for EMA, 20 for vol MA)
    start_idx = max(35, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or
            np.isnan(ema_slope_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: at least 1.5x average
        vol_spike = volume[i] > vol_ma[i] * 1.5
        
        if position == 0:
            # Long: price breaks above S1 with bullish trend and volume
            if close[i] > camarilla_s1_aligned[i] and ema_slope_aligned[i] > 0 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below R1 with bearish trend and volume
            elif close[i] < camarilla_r1_aligned[i] and ema_slope_aligned[i] < 0 and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price crosses below R1 or trend turns bearish
                if close[i] < camarilla_r1_aligned[i] or ema_slope_aligned[i] < 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price crosses above S1 or trend turns bullish
                if close[i] > camarilla_s1_aligned[i] or ema_slope_aligned[i] > 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals