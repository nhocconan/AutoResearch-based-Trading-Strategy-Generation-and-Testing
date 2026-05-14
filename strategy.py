#!/usr/bin/env python3
# Hypothesis: 1h strategy using 4h Camarilla R1/S1 breakout with 1d EMA50 trend filter and 4h volume confirmation (>1.5x 20-period average).
# Long when price breaks above R1 AND close > 1d EMA50 (bullish trend) AND volume > 1.5x 20-period average.
# Short when price breaks below S1 AND close < 1d EMA50 (bearish trend) AND volume > 1.5x 20-period average.
# Exit when price retests the 1d EMA50 level (mean reversion to trend) or opposite Camarilla level touched.
# Uses 1d HTF for trend to reduce noise and overtrading vs shorter trends. Volume confirmation (1.5x) reduces false signals.
# Target: 80-160 total trades over 4 years (20-40/year) to stay within fee drag limits for 1h timeframe.
# Camarilla pivot levels provide high-probability reversal/breakout levels, effective in both bull and bear markets when combined with HTF trend filter.

name = "1h_Camarilla_R1S1_Breakout_1dEMA50_4hVolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 4h Indicators (MTF) ---
    # 4h volume confirmation: > 1.5x 20-period average (balanced filter to control trades)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    volume_4h = df_4h['volume'].values
    vol_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_confirm_4h_raw = volume_4h > (1.5 * vol_ma_20_4h)
    volume_confirm_4h = align_htf_to_ltf(prices, df_4h, volume_confirm_4h_raw)
    
    # --- 4h Camarilla Pivot Levels (R1, S1) ---
    # Calculate from previous 4h bar's OHLC
    prev_close_4h = np.roll(df_4h['close'].values, 1)
    prev_high_4h = np.roll(df_4h['high'].values, 1)
    prev_low_4h = np.roll(df_4h['low'].values, 1)
    prev_close_4h[0] = np.nan
    prev_high_4h[0] = np.nan
    prev_low_4h[0] = np.nan
    
    pivot_4h = (prev_high_4h + prev_low_4h + prev_close_4h) / 3.0
    range_4h = prev_high_4h - prev_low_4h
    r1_4h = pivot_4h + (range_4h * 1.1 / 4.0)  # R1 = pivot + (high-low)*1.1/4
    s1_4h = pivot_4h - (range_4h * 1.1 / 4.0)  # S1 = pivot - (high-low)*1.1/4
    
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) - trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(r1_4h_aligned[i]) or
            np.isnan(s1_4h_aligned[i]) or
            np.isnan(volume_confirm_4h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R1 AND close > 1d EMA50 (bullish trend) AND volume confirm
            if (close[i] > r1_4h_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_confirm_4h[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below S1 AND close < 1d EMA50 (bearish trend) AND volume confirm
            elif (close[i] < s1_4h_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_confirm_4h[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price retests 1d EMA50 (mean reversion to trend) OR touches S1 (opposite level)
            if (close[i] <= ema_50_1d_aligned[i] or 
                close[i] < s1_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price retests 1d EMA50 (mean reversion to trend) OR touches R1 (opposite level)
            if (close[i] >= ema_50_1d_aligned[i] or 
                close[i] > r1_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals