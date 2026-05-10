#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hTrend_Volume_v2
Hypothesis: Breakout above Camarilla R1 or below S1 with 12h EMA50 trend filter and volume confirmation.
Uses volume-adjusted breakout thresholds and adaptive position sizing to reduce whipsaw in ranging markets.
Designed to work in both bull and bear markets by following 12h trend and using volatility-adjusted breakouts.
Target: 20-30 trades/year per symbol with strict entry conditions to minimize fee drag.
"""

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_Volume_v2"
timeframe = "4h"
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
    
    # Calculate 12h EMA50 for trend filter (using HTF data)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 50:
        ema_50_12h[49] = np.mean(close_12h[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_12h)):
            ema_50_12h[i] = alpha * close_12h[i] + (1 - alpha) * ema_50_12h[i-1]
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(21, 50)  # Ensure volume SMA and EMA are ready
    
    for i in range(start_idx, n):
        if np.isnan(atr[i]) or np.isnan(vol_sma[i]) or np.isnan(ema_50_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla levels from previous day
        if i >= 6:  # Need at least 6 bars (previous day's data)
            # Previous day's OHLC (assuming 6 bars per day on 4h chart)
            prev_day_start = max(0, i - 6)
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
            if close[i] > r1 and close[i] > ema_50_12h_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Break below S1 with downtrend and volume confirmation
            elif close[i] < s1 and close[i] < ema_50_12h_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Close crosses back below 12h EMA50
            if close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Close crosses back above 12h EMA50
            if close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals