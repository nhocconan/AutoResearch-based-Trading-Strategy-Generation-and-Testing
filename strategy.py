#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSpike
# Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume spike.
# Long: price breaks above R1, 1d EMA34 rising, volume > 1.8x 20-period average.
# Short: price breaks below S1, 1d EMA34 falling, volume > 1.8x 20-period average.
# Exit: price returns to H4/L4 or 1d EMA34 trend reverses.
# Camarilla levels provide intraday support/resistance, EMA34 filters trend, volume confirms strength.
# Designed for 4-8 trades per month (~48-96/year) to minimize fee drag while capturing significant moves.

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSpike"
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
    
    # --- Camarilla levels from previous day (H4, L4, R1, S1) ---
    H4 = np.full(n, np.nan)
    L4 = np.full(n, np.nan)
    R1 = np.full(n, np.nan)
    S1 = np.full(n, np.nan)
    for i in range(1, n):
        # Use previous day's OHLC (assuming daily data aligned to 4h)
        # Since we're on 4h timeframe, we need to get daily OHLC from 1d data
        pass  # Will calculate from 1d data below
    
    # Calculate Camarilla levels from 1d OHLC
    # Camarilla: H4 = close + 1.5*(high-low), L4 = close - 1.5*(high-low)
    # R1 = close + 1.1*(high-low), S1 = close - 1.1*(high-low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate levels for each 1d bar
    H4_1d = close_1d + 1.5 * (high_1d - low_1d)
    L4_1d = close_1d - 1.5 * (high_1d - low_1d)
    R1_1d = close_1d + 1.1 * (high_1d - low_1d)
    S1_1d = close_1d - 1.1 * (high_1d - low_1d)
    
    # Align 1d levels to 4h timeframe (each 1d bar = 6 four-hour bars)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4_1d)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4_1d)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1_1d)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1_1d)
    
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
    
    # Align EMA and slope to 4h
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    ema_slope_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_slope_1d)
    
    # --- Volume confirmation (volume > 20-period average) ---
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for Camarilla (needs 1d data), EMA34, and volume MA(20)
    start_idx = max(1, 34, 20)  # Need at least 1 day of data
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(R1_aligned[i]) or
            np.isnan(S1_aligned[i]) or
            np.isnan(L4_aligned[i]) or
            np.isnan(H4_aligned[i]) or
            np.isnan(ema_1d_aligned[i]) or
            np.isnan(ema_slope_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Camarilla breakout conditions
        breakout_up = close[i] > R1_aligned[i]
        breakout_down = close[i] < S1_aligned[i]
        
        # Volume spike condition
        vol_spike = volume[i] > vol_ma[i] * 1.8  # 80% above average
        
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
                # Exit long: price falls to L4 OR EMA34 slope turns negative
                if close[i] < L4_aligned[i] or ema_slope_1d_aligned[i] < 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price rises to H4 OR EMA34 slope turns positive
                if close[i] > H4_aligned[i] or ema_slope_1d_aligned[i] > 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals