#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Camarilla pivot R1/S1 breakout on 12h timeframe, filtered by 1d EMA34 trend and volume spike.
# Long when: price breaks above R1, 1d EMA34 rising, volume > 1.5x 20-period average.
# Short when: price breaks below S1, 1d EMA34 falling, volume > 1.5x 20-period average.
# Exit when price returns to Camarilla pivot (P) or 1d EMA34 trend reverses.
# Camarilla levels provide high-probability reversal points, EMA34 filters counter-trend moves, volume confirms breakout strength.
# Works in bull markets by catching R1 breakouts and in bear by catching S1 breakdowns.

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    # Get 1d data for Camarilla pivots and EMA34 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Camarilla pivot levels from 1d (using previous day's OHLC) ---
    # Calculate for each 1d bar, then shift by 1 to avoid look-ahead (use previous day's data)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price for pivot calculation
    typical_price_1d = (high_1d + low_1d + close_1d) / 3
    # Camarilla levels
    P = (high_1d + low_1d + close_1d) / 3  # Pivot point
    R1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    S1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Shift by 1 to use previous day's levels (avoid look-ahead)
    P = np.roll(P, 1)
    R1 = np.roll(R1, 1)
    S1 = np.roll(S1, 1)
    P[0] = np.nan
    R1[0] = np.nan
    S1[0] = np.nan
    
    # --- 1d EMA34 trend ---
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
    
    # --- Volume confirmation (volume > 20-period average) ---
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align 1d indicators to 12h
    P_aligned = align_htf_to_ltf(prices, df_1d, P)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    ema_slope_aligned = align_htf_to_ltf(prices, df_1d, ema_slope)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for EMA34 and volume MA(20)
    start_idx = max(35, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(R1_aligned[i]) or
            np.isnan(S1_aligned[i]) or
            np.isnan(P_aligned[i]) or
            np.isnan(ema_1d_aligned[i]) or
            np.isnan(ema_slope_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Camarilla breakout conditions
        breakout_up = close[i] > R1_aligned[i]
        breakout_down = close[i] < S1_aligned[i]
        
        # Volume spike condition
        vol_spike = volume[i] > vol_ma[i] * 1.5  # 50% above average
        
        if position == 0:
            if breakout_up and ema_slope_aligned[i] > 0 and vol_spike:
                # Long: upward breakout above R1 + rising EMA34 + volume spike
                signals[i] = 0.25
                position = 1
            elif breakout_down and ema_slope_aligned[i] < 0 and vol_spike:
                # Short: downward breakout below S1 + falling EMA34 + volume spike
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price falls to pivot P OR EMA34 slope turns negative
                if close[i] < P_aligned[i] or ema_slope_aligned[i] < 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price rises to pivot P OR EMA34 slope turns positive
                if close[i] > P_aligned[i] or ema_slope_aligned[i] > 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals