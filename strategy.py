#!/usr/bin/env python3
"""
12h_WVWAP_MeanReversion_Bands
Hypothesis: 12-hour price reverts to volume-weighted moving average (VWAP) with Bollinger Bands as dynamic thresholds.
In ranging markets (common in 2025+), price tends to revert to VWAP. Bands adapt to volatility, reducing whipsaws.
Works in both bull/bear: mean reversion is stronger in ranging/low-volatility regimes, which occur in all market phases.
Uses 1-day trend filter to avoid counter-trend trades during strong moves. Target: 15-25 trades/year.
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
    
    # Calculate 12-hour VWAP (typical price * volume cumulative)
    typical_price = (high + low + close) / 3.0
    pv = typical_price * volume
    cum_pv = np.cumsum(pv)
    cum_vol = np.cumsum(volume)
    vwap_12h_raw = cum_pv / cum_vol
    
    # Get actual 12h data for proper alignment
    df_12h = get_htf_data(prices, '12h')
    # Resample VWAP to 12h by taking last value of each 12h bar
    # We'll compute VWAP on 12h data directly for cleaner alignment
    typical_price_12h = (df_12h['high'] + df_12h['low'] + df_12h['close']) / 3.0
    pv_12h = typical_price_12h * df_12h['volume'].values
    cum_pv_12h = np.cumsum(pv_12h)
    cum_vol_12h = np.cumsum(df_12h['volume'].values)
    vwap_12h = cum_pv_12h / cum_vol_12h
    
    # Align 12h VWAP to 12h timeframe (no additional delay needed as VWAP is cumulative)
    vwap_12h_aligned = align_htf_to_ltf(prices, df_12h, vwap_12h)
    
    # Calculate 12h rolling standard deviation for Bollinger Bands
    # Use typical price for volatility calculation
    tp_12h = typical_price_12h
    tp_mean = np.full_like(tp_12h, np.nan)
    tp_std = np.full_like(tp_12h, np.nan)
    
    # Calculate rolling mean and std with min_periods=20
    for i in range(len(tp_12h)):
        if i >= 19:  # 20-period window
            window = tp_12h[i-19:i+1]
            tp_mean[i] = np.mean(window)
            tp_std[i] = np.std(window)
    
    # Bollinger Bands: ±2 std dev
    upper_bb = tp_mean + 2 * tp_std
    lower_band = tp_mean - 2 * tp_std
    
    # Align bands to 12h timeframe
    upper_bb_aligned = align_htf_to_ltf(prices, df_12h, upper_bb)
    lower_band_aligned = align_htf_to_ltf(prices, df_12h, lower_band)
    
    # 1-day trend filter: EMA50
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema50_1d[49] = np.mean(close_1d[0:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema50_1d[i] = close_1d[i] * alpha + ema50_1d[i-1] * (1 - alpha)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Entry conditions
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for EMA50 and Bollinger Bands
    
    for i in range(start_idx, n):
        if (np.isnan(vwap_12h_aligned[i]) or np.isnan(upper_bb_aligned[i]) or 
            np.isnan(lower_band_aligned[i]) or np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price touches or goes below lower band, but 1d trend is up (avoid strong downtrends)
            if close[i] <= lower_band_aligned[i] and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price touches or goes above upper band, but 1d trend is down (avoid strong uptrends)
            elif close[i] >= upper_bb_aligned[i] and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to VWAP or breaks above upper band (momentum)
            if close[i] >= vwap_12h_aligned[i] or close[i] >= upper_bb_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to VWAP or breaks below lower band (momentum)
            if close[i] <= vwap_12h_aligned[i] or close[i] <= lower_band_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WVWAP_MeanReversion_Bands"
timeframe = "12h"
leverage = 1.0