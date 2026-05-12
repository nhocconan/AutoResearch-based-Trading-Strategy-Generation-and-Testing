#!/usr/bin/env python3
"""
4h Camarilla R1/S1 Breakout with 12h EMA Trend and Volume Spike
Hypothesis: Camarilla pivot levels (R1/S1) act as significant support/resistance.
Breakouts above R1 with bullish 12h EMA trend and volume confirmation capture strong moves.
Breakdowns below S1 with bearish 12h EMA trend and volume confirmation capture reversals.
Designed for low trade frequency (20-50/year) to minimize fee drift while capturing
trend continuation and reversal moves in both bull and bear markets.
"""
name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeS"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Camarilla Pivot Levels from previous day ===
    # Use daily high, low, close from previous day
    # We'll calculate daily OHLC from 1d data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day
    # R1 = C + (H-L)*1.1/12
    # S1 = C - (H-L)*1.1/12
    # Where C, H, L are previous day's close, high, low
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align to 4h timeframe (wait for daily bar to close)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # === 12h EMA50 for trend filter ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # === Volume Spike (20-period) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)  # Require 2x average volume
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R1 + 12h EMA50 rising + volume spike
            if (close[i] > camarilla_r1_aligned[i] and 
                ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1] and
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 + 12h EMA50 falling + volume spike
            elif (close[i] < camarilla_s1_aligned[i] and 
                  ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1] and
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price breaks below S1 OR 12h EMA50 turns down
            if close[i] < camarilla_s1_aligned[i] or ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R1 OR 12h EMA50 turns up
            if close[i] > camarilla_r1_aligned[i] or ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals