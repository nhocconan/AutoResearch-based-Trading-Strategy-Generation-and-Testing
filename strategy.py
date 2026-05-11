#!/usr/bin/env python3
"""
4h_1d_Camarilla_R1_S1_Breakout_TrendFilter_Volume_v2
Hypothesis: Tightening the original strategy by requiring stronger volume confirmation (2.0x average) and adding a 4h ADX filter (ADX>25) to ensure trades occur only in trending markets. This reduces trade frequency while maintaining edge in both bull and bear markets by following the 1d trend with confirmed momentum.
"""

name = "4h_1d_Camarilla_R1_S1_Breakout_TrendFilter_Volume_v2"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d OHLCV for Camarilla Pivot Levels ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate pivot points using previous 1d's OHLC
    prev_high_1d = df_1d['high'].values
    prev_low_1d = df_1d['low'].values
    prev_close_1d = df_1d['close'].values
    
    pivot_1d = (prev_high_1d + prev_low_1d + prev_close_1d) / 3.0
    range_val_1d = prev_high_1d - prev_low_1d
    
    # Camarilla levels (R1 and S1)
    R1_1d = pivot_1d + (range_val_1d * 1.1 / 12)
    S1_1d = pivot_1d - (range_val_1d * 1.1 / 12)
    
    # Align to 4h timeframe
    R1_4h = align_htf_to_ltf(prices, df_1d, R1_1d)
    S1_4h = align_htf_to_ltf(prices, df_1d, S1_1d)
    
    # --- 1d EMA21 Trend Filter ---
    ema_21_1d = pd.Series(df_1d['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    
    # --- 4h ADX for trend strength filter ---
    # Calculate TR (True Range)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # First period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate +DM and -DM
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: smoothed = previous * (1 - 1/period) + current * (1/period)
        for i in range(period, len(data)):
            result[i] = result[i-1] * (1 - 1/period) + data[i] * (1/period)
        return result
    
    atr_period = 14
    tr_smooth = wilder_smooth(tr, atr_period)
    plus_dm_smooth = wilder_smooth(plus_dm, atr_period)
    minus_dm_smooth = wilder_smooth(minus_dm, atr_period)
    
    # Avoid division by zero
    plus_di = np.where(tr_smooth != 0, 100 * plus_dm_smooth / tr_smooth, 0)
    minus_di = np.where(tr_smooth != 0, 100 * minus_dm_smooth / tr_smooth, 0)
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = wilder_smooth(dx, atr_period)
    
    # --- Volume Spike Detection (12-period average on 4h) ---
    vol_ma = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for ADX and other indicators)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN (first few bars)
        if (np.isnan(R1_4h[i]) or np.isnan(S1_4h[i]) or 
            np.isnan(ema_21_4h[i]) or np.isnan(adx[i]) or np.isnan(vol_ratio[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation threshold (stronger than before)
        volume_spike = vol_ratio[i] > 2.0
        # Trend strength filter
        strong_trend = adx[i] > 25
        
        if position == 0:
            # Long: price breaks above R1 with volume, above EMA21, in strong trend
            if (close[i] > R1_4h[i] and 
                volume_spike and 
                close[i] > ema_21_4h[i] and
                strong_trend):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume, below EMA21, in strong trend
            elif (close[i] < S1_4h[i] and 
                  volume_spike and 
                  close[i] < ema_21_4h[i] and
                  strong_trend):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite breakout or loss of momentum/trend
            if position == 1:
                # Exit long: price breaks below S1 (reversal signal) OR trend weakens
                if close[i] < S1_4h[i] or adx[i] < 20:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price breaks above R1 (reversal signal) OR trend weakens
                if close[i] > R1_4h[i] or adx[i] < 20:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals