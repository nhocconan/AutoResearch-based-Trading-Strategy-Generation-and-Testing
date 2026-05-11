#!/usr/bin/env python3
# 4h_CamR1S1_Breakout_1dEMA34_Volume_Regime
# Hypothesis: Camarilla pivot (R1/S1) breakout on 4h with 1d EMA34 trend filter and volume confirmation.
# Long: price closes above R1, EMA34 rising, volume > 1.5x avg. Short: price closes below S1, EMA34 falling, volume > 1.5x avg.
# Exit: price crosses back to pivot point (PP) or EMA34 trend reverses.
# Camarilla provides intraday support/resistance, EMA34 filters trend, volume confirms breakout strength.
# Works in bull/bear by only taking breaks in trend direction, reducing whipsaw.

name = "4h_CamR1S1_Breakout_1dEMA34_Volume_Regime"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for pivot and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Camarilla levels (from previous 1d) ---
    # Using previous day's H, L, C to avoid look-ahead
    ph = df_1d['high'].values
    pl = df_1d['low'].values
    pc = df_1d['close'].values
    
    # Calculate Camarilla for each 1d bar
    R1 = np.full(len(ph), np.nan)
    S1 = np.full(len(ph), np.nan)
    PP = np.full(len(ph), np.nan)
    
    for i in range(len(ph)):
        if i == 0:
            # First day: use same values (no prior day)
            R1[i] = pc[i] + (ph[i] - pl[i]) * 1.1 / 12
            S1[i] = pc[i] - (ph[i] - pl[i]) * 1.1 / 12
            PP[i] = (ph[i] + pl[i] + pc[i]) / 3
        else:
            # Use previous day's OHLC
            R1[i] = pc[i-1] + (ph[i-1] - pl[i-1]) * 1.1 / 12
            S1[i] = pc[i-1] - (ph[i-1] - pl[i-1]) * 1.1 / 12
            PP[i] = (ph[i-1] + pl[i-1] + pc[i-1]) / 3
    
    # --- 1d EMA34 trend ---
    close_1d = df_1d['close'].values
    ema_1d = np.full(len(close_1d), np.nan)
    for i in range(34, len(close_1d)):
        if i == 34:
            ema_1d[i] = np.mean(close_1d[0:34])
        else:
            ema_1d[i] = (close_1d[i] * 2 / (34 + 1)) + (ema_1d[i-1] * (33 / (34 + 1)))
    
    # EMA slope
    ema_slope = np.full(len(close_1d), np.nan)
    for i in range(35, len(close_1d)):
        ema_slope[i] = ema_1d[i] - ema_1d[i-1]
    
    # --- Volume confirmation ---
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align 1d data to 4h
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    PP_aligned = align_htf_to_ltf(prices, df_1d, PP)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    ema_slope_aligned = align_htf_to_ltf(prices, df_1d, ema_slope)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for Camarilla (need prev day), EMA34, volume MA
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(R1_aligned[i]) or
            np.isnan(S1_aligned[i]) or
            np.isnan(PP_aligned[i]) or
            np.isnan(ema_1d_aligned[i]) or
            np.isnan(ema_slope_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike condition
        vol_spike = volume[i] > vol_ma[i] * 1.5
        
        if position == 0:
            if close[i] > R1_aligned[i] and ema_slope_aligned[i] > 0 and vol_spike:
                # Long: break above R1, rising EMA, volume spike
                signals[i] = 0.25
                position = 1
            elif close[i] < S1_aligned[i] and ema_slope_aligned[i] < 0 and vol_spike:
                # Short: break below S1, falling EMA, volume spike
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price crosses below PP or EMA turns down
                if close[i] < PP_aligned[i] or ema_slope_aligned[i] < 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price crosses above PP or EMA turns up
                if close[i] > PP_aligned[i] or ema_slope_aligned[i] > 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals