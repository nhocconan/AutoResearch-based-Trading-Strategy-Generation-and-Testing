#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: 12h Camarilla R1/S1 breakout for momentum, filtered by 1d EMA34 trend and volume spike.
# Long when: price breaks above Camarilla R1, 1d EMA34 rising, volume > 1.5x 20-period avg.
# Short when: price breaks below Camarilla S1, 1d EMA34 falling, volume > 1.5x 20-period avg.
# Exit when price crosses back to Camarilla midpoint (C) or 1d EMA34 trend reverses.
# Works in bull markets by catching breakouts and in bear by catching breakdowns with trend filter.
# Camarilla provides clear structure, EMA34 filters counter-trend moves, volume confirms strength.
# Target: 12h timeframe, 12-37 trades/year, low frequency to avoid fee drag.

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
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
    
    # --- Camarilla R1, S1, C from previous day (using daily high/low/close) ---
    # Camarilla levels calculated from previous day's range
    camarilla_R1 = np.full(n, np.nan)
    camarilla_S1 = np.full(n, np.nan)
    camarilla_C = np.full(n, np.nan)
    
    # Calculate daily range from 1d data
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    daily_range = daily_high - daily_low
    
    # Camarilla formulas: R1 = close + (range * 1.1/12), S1 = close - (range * 1.1/12), C = close
    camarilla_R1_1d = daily_close + (daily_range * 1.1 / 12)
    camarilla_S1_1d = daily_close - (daily_range * 1.1 / 12)
    camarilla_C_1d = daily_close
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1_1d)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1_1d)
    camarilla_C_aligned = align_htf_to_ltf(prices, df_1d, camarilla_C_1d)
    
    # --- 1d EMA34 trend ---
    close_1d = df_1d['close'].values
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
    
    # Warmup: enough for Camarilla (need 1d data), EMA34, and volume MA(20)
    start_idx = max(20, 35, 20)  # need at least 35 for EMA slope
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(camarilla_R1_aligned[i]) or
            np.isnan(camarilla_S1_aligned[i]) or
            np.isnan(camarilla_C_aligned[i]) or
            np.isnan(ema_1d_aligned[i]) or
            np.isnan(ema_slope_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Camarilla breakout conditions
        breakout_up = close[i] > camarilla_R1_aligned[i]
        breakout_down = close[i] < camarilla_S1_aligned[i]
        
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
                # Exit long: price falls to Camarilla C (midpoint) OR EMA34 slope turns negative
                if close[i] < camarilla_C_aligned[i] or ema_slope_1d_aligned[i] < 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price rises to Camarilla C (midpoint) OR EMA34 slope turns positive
                if close[i] > camarilla_C_aligned[i] or ema_slope_1d_aligned[i] > 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals