#!/usr/bin/env python3
"""
4h_WideRangeBreakout_VolumeTrend
Hypothesis: Trade wide-range 4h breakouts (>2x ATR range) in the direction of 1d EMA trend, confirmed by volume >2x average. Uses 1d EMA as trend filter to avoid counter-trend trades. Wide-range breakouts capture momentum after volatility expansion, which works in both bull and bear markets. Position size 0.25, max 40 trades/year to avoid fee drag.
"""

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
    
    # Get 4h data for breakout detection
    df_4h = get_htf_data(prices, '4h')
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 4h calculations
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Previous 4h bar's OHLC (completed bar)
    prev_high_4h = np.roll(high_4h, 1)
    prev_low_4h = np.roll(low_4h, 1)
    prev_close_4h = np.roll(close_4h, 1)
    prev_high_4h[0] = high_4h[0]
    prev_low_4h[0] = low_4h[0]
    prev_close_4h[0] = close_4h[0]
    
    # 4h range and ATR(20)
    range_4h = prev_high_4h - prev_low_4h
    atr_mult = 2.0
    atr_period = 20
    
    # True Range for ATR
    tr1 = np.abs(high_4h - low_4h)
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high_4h[0] - low_4h[0]
    
    # ATR calculation with smoothing
    atr = np.zeros_like(tr)
    if len(tr) >= atr_period:
        atr[atr_period] = np.mean(tr[1:atr_period+1])
        for i in range(atr_period + 1, len(tr)):
            atr[i] = (atr[i-1] * (atr_period - 1) + tr[i]) / atr_period
    
    # Breakout levels: break above/below previous 4h range by atr_mult * ATR
    breakout_up = prev_high_4h + atr_mult * atr
    breakout_down = prev_low_4h - atr_mult * atr
    
    # 1d EMA trend filter
    close_1d = df_1d['close'].values
    ema_period = 50
    if len(close_1d) >= ema_period:
        ema_1d = np.zeros_like(close_1d)
        ema_1d[ema_period-1] = np.mean(close_1d[:ema_period])
        for i in range(ema_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 / (ema_period + 1)) + (ema_1d[i-1] * (ema_period - 1) / (ema_period + 1))
    else:
        ema_1d = np.full_like(close_1d, np.nan)
    
    # Align higher timeframe data to 4h
    breakout_up_aligned = align_htf_to_ltf(prices, df_4h, breakout_up)
    breakout_down_aligned = align_htf_to_ltf(prices, df_4h, breakout_down)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr)
    
    # Volume confirmation: volume > 2x 20-period average
    vol_ma = np.zeros_like(volume)
    vol_period = 20
    for i in range(vol_period, len(volume)):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, vol_period, atr_period, ema_period)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(breakout_up_aligned[i]) or np.isnan(breakout_down_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(atr_4h_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above breakout_up with volume and above 1d EMA
            if close[i] > breakout_up_aligned[i] and vol_confirm and close[i] > ema_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below breakout_down with volume and below 1d EMA
            elif close[i] < breakout_down_aligned[i] and vol_confirm and close[i] < ema_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price closes below breakout_down (reverse signal) or below 1d EMA
            if close[i] < breakout_down_aligned[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above breakout_up (reverse signal) or above 1d EMA
            if close[i] > breakout_up_aligned[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WideRangeBreakout_VolumeTrend"
timeframe = "4h"
leverage = 1.0