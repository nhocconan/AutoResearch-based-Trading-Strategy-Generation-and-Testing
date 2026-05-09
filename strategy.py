#!/usr/bin/env python3
# 4H_1D_RVOL_MeanReversion_WithTrend
# Hypothesis: Mean reversion in 4h works when price deviates significantly from 1d VWAP, but only in the direction of 1d trend to avoid counter-trend trades.
# Uses 1d VWAP as fair value, 4h RVO (Relative Volume) for confirmation, and 1d EMA50 for trend filter.
# Designed for both bull and bear markets: in uptrend, buy dips to VWAP; in downtrend, sell rallies to VWAP.
# Target: 20-40 trades/year per symbol (80-160 total over 4 years) to avoid fee drag.

name = "4H_1D_RVOL_MeanReversion_WithTrend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for VWAP and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d VWAP (volume-weighted average price)
    typical_price_1d = (high_1d + low_1d + close_1d) / 3
    vwap_1d = (typical_price_1d * volume_1d).cumsum() / volume_1d.cumsum()
    
    # 1d trend: EMA(50) on close
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_up = close_1d > ema_50
    
    # 4h Relative Volume: current volume / 20-period average volume
    volume_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    rvol = volume / np.where(volume_avg_20 > 0, volume_avg_20, 1)  # avoid div by zero
    
    # Align 1d indicators to 4h
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, trend_up)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(vwap_1d_aligned[i]) or np.isnan(trend_up_aligned[i]) or np.isnan(rvol[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Mean reversion triggers: price deviates >1.5% from 1d VWAP
        dev_pct = (close[i] - vwap_1d_aligned[i]) / vwap_1d_aligned[i] * 100
        
        if position == 0:
            # Enter long: price below VWAP (oversold) + 1d uptrend + volume confirmation (RVO > 1.5)
            if dev_pct < -1.5 and trend_up_aligned[i] and rvol[i] > 1.5:
                signals[i] = 0.25
                position = 1
            # Enter short: price above VWAP (overbought) + 1d downtrend + volume confirmation (RVO > 1.5)
            elif dev_pct > 1.5 and not trend_up_aligned[i] and rvol[i] > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to VWAP or trend changes
            if dev_pct > -0.5 or not trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to VWAP or trend changes
            if dev_pct < 0.5 or trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals