#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: 12h Camarilla pivot breakout for momentum, filtered by 1d EMA34 trend and volume spike.
# Long when: price breaks above Camarilla R1, 1d EMA34 rising, volume > 1.5x 20-period avg.
# Short when: price breaks below Camarilla S1, 1d EMA34 falling, volume > 1.5x 20-period avg.
# Exit when price crosses back to Camarilla midpoint or 1d EMA34 trend reverses.
# Works in bull markets by catching breakouts and in bear by catching breakdowns with trend filter.
# Camarilla provides clear support/resistance levels, EMA34 filters counter-trend moves, volume confirms strength.

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 34:
        return np.zeros(n)
    
    # Get 1d data for EMA34 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Camarilla pivot levels from previous day ---
    # Calculate from previous day's OHLC (1d data)
    camarilla_r1 = np.full(n, np.nan)
    camarilla_s1 = np.full(n, np.nan)
    camarilla_mid = np.full(n, np.nan)
    
    for i in range(1, n):
        # Use previous day's data (1d timeframe)
        # For 12h bar at index i, we need the completed 1d bar that ended before it
        # We'll use the 1d data shifted by 1 bar to avoid lookahead
        pass  # Will fill in after calculating daily values
    
    # Calculate daily Camarilla levels from 1d OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r1_1d = np.full(len(close_1d), np.nan)
    camarilla_s1_1d = np.full(len(close_1d), np.nan)
    camarilla_mid_1d = np.full(len(close_1d), np.nan)
    
    for i in range(len(close_1d)):
        # Camarilla levels: based on previous day's range
        if i > 0:  # Need previous day's data
            prev_high = high_1d[i-1]
            prev_low = low_1d[i-1]
            prev_close = close_1d[i-1]
            range_val = prev_high - prev_low
            camarilla_mid_1d[i] = (prev_high + prev_low) / 2
            camarilla_r1_1d[i] = camarilla_mid_1d[i] + (range_val * 1.1 / 12)
            camarilla_s1_1d[i] = camarilla_mid_1d[i] - (range_val * 1.1 / 12)
        # For first bar, no previous day data available
    
    # Align 1d Camarilla levels to 12h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1_1d)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1_1d)
    camarilla_mid_aligned = align_htf_to_ltf(prices, df_1d, camarilla_mid_1d)
    
    # --- 1d EMA34 trend ---
    ema_1d = np.full(len(close_1d), np.nan)
    for i in range(34, len(close_1d)):
        if i == 34:
            ema_1d[i] = np.mean(close_1d[0:34])
        else:
            ema_1d[i] = (close_1d[i] * 2 / (34 + 1)) + (ema_1d[i-1] * (33 / (34 + 1)))
    
    # EMA slope (rising/falling)
    ema_slope_1d = np.full(len(close_1d), np.nan)
    for i in range(35, len(close_1d)):
        ema_slope_1d[i] = ema_1d[i] - ema_1d[i-1]
    
    # Align 1d EMA and slope to 12h
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    ema_slope_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_slope_1d)
    
    # --- Volume confirmation (volume > 20-period average) ---
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for Camarilla (need 2 days), EMA34, and volume MA(20)
    start_idx = max(20, 35, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(camarilla_mid_aligned[i]) or
            np.isnan(ema_1d_aligned[i]) or
            np.isnan(ema_slope_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Camarilla breakout conditions
        breakout_up = close[i] > camarilla_r1_aligned[i]
        breakout_down = close[i] < camarilla_s1_aligned[i]
        
        # Volume spike condition
        vol_spike = volume[i] > vol_ma[i] * 1.5  # 50% above average
        
        if position == 0:
            if breakout_up and ema_slope_1d_aligned[i] > 0 and vol_spike:
                # Long: upward breakout + rising EMA34 + volume spike
                signals[i] = 0.25
                position = 1
            elif breakout_down and ema_slope_1d_aligned[i] < 0 and vol_spike:
                # Short: downward breakout + falling EMA34 + volume spike
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price falls to midpoint OR EMA34 slope turns negative
                if close[i] < camarilla_mid_aligned[i] or ema_slope_1d_aligned[i] < 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price rises to midpoint OR EMA34 slope turns positive
                if close[i] > camarilla_mid_aligned[i] or ema_slope_1d_aligned[i] > 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals