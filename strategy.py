#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_Volume
Hypothesis: Breakout above Camarilla R1 or below S1 with 1d EMA34 trend filter and volume confirmation.
Designed to work in both bull and bear markets by following 1d trend and using volatility-adjusted breakouts.
Target: 15-25 trades/year per symbol with strict entry conditions to minimize fee drag.
"""

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for volatility measurement
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = np.inf
    tr3[0] = np.inf
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            atr[i] = np.mean(tr[1:15])
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Calculate volume SMA(20) for volume filter
    vol_sma = np.full(n, np.nan)
    for i in range(20, n):
        vol_sma[i] = np.mean(volume[i-20:i])
    
    # Calculate 1d EMA34 for trend filter (using HTF data)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        ema_34_1d[33] = np.mean(close_1d[:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1d)):
            ema_34_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_34_1d[i-1]
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(21, 34)  # Ensure volume SMA and EMA are ready
    
    for i in range(start_idx, n):
        if np.isnan(atr[i]) or np.isnan(vol_sma[i]) or np.isnan(ema_34_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla levels from previous day
        if i >= 2:  # Need at least 2 bars (previous day's data for 12h chart)
            # Previous day's OHLC (assuming 2 bars per day on 12h chart)
            prev_day_start = max(0, i - 2)
            prev_day_high = np.max(high[prev_day_start:i])
            prev_day_low = np.min(low[prev_day_start:i])
            prev_day_close = close[i-1]
            
            # Camarilla levels
            range_val = prev_day_high - prev_day_low
            # Use volatility-adjusted breakout thresholds
            volatility_factor = np.clip(atr[i] / (prev_day_high - prev_day_low + 1e-10), 0.5, 2.0)
            r1 = prev_day_close + (range_val * 1.1 / 12) * volatility_factor
            s1 = prev_day_close - (range_val * 1.1 / 12) * volatility_factor
        else:
            # Not enough data for Camarilla calculation
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.8x average volume (stricter)
        volume_confirm = volume[i] > 1.8 * vol_sma[i]
        
        if position == 0:
            # Long: Break above R1 with uptrend and volume confirmation
            if close[i] > r1 and close[i] > ema_34_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Break below S1 with downtrend and volume confirmation
            elif close[i] < s1 and close[i] < ema_34_1d_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Close crosses back below 1d EMA34
            if close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Close crosses back above 1d EMA34
            if close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals