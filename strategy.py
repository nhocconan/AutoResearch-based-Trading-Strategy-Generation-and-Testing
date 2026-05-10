#!/usr/bin/env python3
"""
6h_Keltner_Channel_MeanReversion
Hypothesis: Mean reversion from Keltner Channel extremes (upper/lower) with 12h trend filter and volume exhaustion filter.
Works in bull/bear by fading extremes only when trend aligns (long in uptrend at lower band, short in downtrend at upper band).
Volume exhaustion filter avoids catching falling knives. Target: 20-30 trades/year (80-120 total).
"""

name = "6h_Keltner_Channel_MeanReversion"
timeframe = "6h"
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
    
    # 12h data for Keltner Channel and trend
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Keltner Channel (20, 2) on 12h
    atr_period = 20
    atr_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= atr_period:
        tr = np.maximum(
            high_12h[1:] - low_12h[1:],
            np.maximum(
                np.abs(high_12h[1:] - close_12h[:-1]),
                np.abs(low_12h[1:] - close_12h[:-1])
            )
        )
        tr = np.concatenate([[np.nan], tr])
        atr_12h[atr_period-1] = np.nanmean(tr[1:atr_period])
        for i in range(atr_period, len(close_12h)):
            atr_12h[i] = (atr_12h[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    ema20_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 20:
        ema20_12h[19] = np.mean(close_12h[:20])
        alpha = 2 / (20 + 1)
        for i in range(20, len(close_12h)):
            ema20_12h[i] = alpha * close_12h[i] + (1 - alpha) * ema20_12h[i-1]
    
    upper_12h = ema20_12h + 2 * atr_12h
    lower_12h = ema20_12h - 2 * atr_12h
    
    # 12h EMA50 for trend filter
    ema50_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 50:
        ema50_12h[49] = np.mean(close_12h[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_12h)):
            ema50_12h[i] = alpha * close_12h[i] + (1 - alpha) * ema50_12h[i-1]
    
    # 12h volume SMA20 for exhaustion filter
    vol_sma20_12h = np.full(len(volume_12h), np.nan)
    if len(volume_12h) >= 20:
        vol_sma20_12h[19] = np.mean(volume_12h[:20])
        for i in range(20, len(volume_12h)):
            vol_sma20_12h[i] = (vol_sma20_12h[i-1] * 19 + volume_12h[i]) / 20
    
    # Align 12h indicators to 6h
    upper_aligned = align_htf_to_ltf(prices, df_12h, upper_12h)
    lower_aligned = align_htf_to_ltf(prices, df_12h, lower_12h)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    vol_sma20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_sma20_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for EMA50
    
    for i in range(start_idx, n):
        if np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_sma20_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume exhaustion: current 6h volume < 0.5x average 12h volume (scaled)
        vol_12h_scaled = vol_sma20_12h_aligned[i] / 4.0  # 4x 6h bars in 12h
        volume_exhausted = volume[i] < 0.5 * vol_12h_scaled
        
        # Trend and price relative to Keltner bands
        is_uptrend = close[i] > ema50_12h_aligned[i]
        is_downtrend = close[i] < ema50_12h_aligned[i]
        price_at_lower = close[i] <= lower_aligned[i]
        price_at_upper = close[i] >= upper_aligned[i]
        
        if position == 0:
            # Long: price at lower band, in uptrend, with volume exhaustion
            if price_at_lower and is_uptrend and volume_exhausted:
                signals[i] = 0.25
                position = 1
            # Short: price at upper band, in downtrend, with volume exhaustion
            elif price_at_upper and is_downtrend and volume_exhausted:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses above EMA20 (mean reversion complete) or trend fails
            if close[i] >= ema20_12h_aligned[i] or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses below EMA20 or trend fails
            if close[i] <= ema20_12h_aligned[i] or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals