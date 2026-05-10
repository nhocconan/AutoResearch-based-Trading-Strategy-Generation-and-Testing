#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dEMA50_Trend_VolumeS
# Hypothesis: Breakout of tighter Camarilla R1/S1 levels with 1-day EMA50 trend filter and volume spike.
# R1/S1 levels are closer to current price than R3/S3, leading to fewer but more significant breakouts.
# EMA50 provides smoother trend filter than EMA34, reducing whipsaw in choppy markets.
# Volume spike (2.5x 20-period EMA) confirms institutional participation.
# Designed for 15-25 trades/year to minimize fee drag while capturing strong moves in bull/bear markets.

name = "4h_Camarilla_R1_S1_Breakout_1dEMA50_Trend_VolumeS"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data for Camarilla and EMA
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla pivot levels from previous day (R1/S1)
    high_prev = df_1d['high'].values
    low_prev = df_1d['low'].values
    close_prev = df_1d['close'].values
    
    # Camarilla formulas for R1/S1 (tighter levels)
    # R1 = C + ((H-L) * 1.0833)
    # S1 = C - ((H-L) * 1.0833)
    camarilla_r1 = close_prev + (high_prev - low_prev) * 1.0833
    camarilla_s1 = close_prev - (high_prev - low_prev) * 1.0833
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Get price, volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 2.5x 20-period EMA (stricter for fewer trades)
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 2.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA50 (50)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above R1 with uptrend and volume
            if close[i] > camarilla_r1_aligned[i] and close[i] > ema_50_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with downtrend and volume
            elif close[i] < camarilla_s1_aligned[i] and close[i] < ema_50_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price back below S1 or trend change
            if close[i] < camarilla_s1_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price back above R1 or trend change
            if close[i] > camarilla_r1_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals