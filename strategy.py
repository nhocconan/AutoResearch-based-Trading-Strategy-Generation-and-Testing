#!/usr/bin/env python3
"""
12h_Midpoint_Reversion_with_Volume_and_1dTrend
Hypothesis: In ranging markets, price tends to revert to the 12h midpoint (average of high-low).
Trades are triggered when price deviates significantly from the midpoint, with volume confirmation
and aligned with the 1d trend (using EMA34). Designed for low-frequency, high-edge setups
on 12h timeframe to avoid overtrading and perform in both bull and bear regimes.
Target: ~20-30 trades/year.
"""

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
    
    # Get 12h data for midpoint calculation
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # 12h midpoint: average of high and low
    midpoint_12h = (high_12h + low_12h) / 2.0
    
    # Deviation from midpoint as percentage of range
    range_12h = high_12h - low_12h
    # Avoid division by zero
    range_12h = np.where(range_12h == 0, 1e-10, range_12h)
    deviation_pct = (close[:len(midpoint_12h)] - midpoint_12h) / range_12h * 100
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # Calculate EMA34 on 1d
    ema34_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        ema34_1d[33] = np.mean(close_1d[0:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1d)):
            ema34_1d[i] = close_1d[i] * alpha + ema34_1d[i-1] * (1 - alpha)
    
    # Volume spike: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    # Align 12h deviation and midpoint to lower timeframe (using close length)
    # We need to map 12h values to each 5m bar (assuming 5m is base, but prices is 12h?)
    # Since timeframe is 12h, prices are already 12h bars
    # So we can use the values directly, but ensure alignment for safety
    deviation_aligned = align_htf_to_ltf(prices, df_12h, deviation_pct)
    midpoint_aligned = align_htf_to_ltf(prices, df_12h, midpoint_12h)
    
    # Align 1d EMA34 to 12h timeframe
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    
    start_idx = max(34, 20)  # warmup for EMA34 and vol MA
    
    for i in range(start_idx, n):
        if (np.isnan(deviation_aligned[i]) or np.isnan(midpoint_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: price deviates >1.5% from midpoint, with volume spike
        # Long when price is significantly below midpoint (oversold in range)
        # Short when price is significantly above midpoint (overbought in range)
        if deviation_aligned[i] < -1.5 and vol_spike[i]:
            # Long bias only if 1d trend is up (price above EMA34)
            if close[i] > ema34_aligned[i]:
                signals[i] = 0.25
        elif deviation_aligned[i] > 1.5 and vol_spike[i]:
            # Short bias only if 1d trend is down (price below EMA34)
            if close[i] < ema34_aligned[i]:
                signals[i] = -0.25
        # Exit when price returns to midpoint (within 0.5%)
        elif abs(deviation_aligned[i]) < 0.5:
            signals[i] = 0.0
        # Otherwise, hold current signal (though we don't track position explicitly,
        # the deviation condition will naturally flip signal when crossing zero)
        # But to avoid whipsaw, we decay to zero if not triggered
        else:
            # Only hold signal if we were just triggered, otherwise zero
            # Simple approach: signal only on trigger bar, then zero
            # This reduces trade frequency
            pass  # already zero by default
    
    return signals

name = "12h_Midpoint_Reversion_with_Volume_and_1dTrend"
timeframe = "12h"
leverage = 1.0