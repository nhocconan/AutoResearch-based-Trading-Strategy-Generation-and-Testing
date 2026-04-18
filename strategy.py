#!/usr/bin/env python3
"""
4h_Trend_1dVWAP_MeanReversion_v1
Hypothesis: Trade 4h trend pullbacks to 1d VWAP. In trending markets (4h EMA50 > EMA200 for long, < for short), price reverts to 1d VWAP, offering low-risk entries. Enter long when 4h EMA50 > EMA200 and price touches or crosses below 1d VWAP; short when EMA50 < EMA200 and price touches or crosses above 1d VWAP. Volume must be > 1.5x 20-period average to confirm institutional interest. Exit when price crosses back over/under 1d VWAP or trend reverses. This captures mean reversion within trends, works in bull/bear by following trend, and limits trades via strict VWAP touch requirement.
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
    
    # Get 1d data for VWAP
    df_1d = get_htf_data(prices, '1d')
    typical_price_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3.0
    vwap_1d = np.full_like(typical_price_1d, np.nan)
    cumulative_tpv = 0.0
    cumulative_volume = 0.0
    for i in range(len(typical_price_1d)):
        cumulative_tpv += typical_price_1d[i] * df_1d['volume'].values[i]
        cumulative_volume += df_1d['volume'].values[i]
        if cumulative_volume > 0:
            vwap_1d[i] = cumulative_tpv / cumulative_volume
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # 4h EMA50 and EMA200 for trend
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    ema200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema200_4h)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, vol_period)
    
    for i in range(start_idx, n):
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(ema200_4h_aligned[i]) or 
            np.isnan(vwap_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: uptrend + price at/below VWAP + volume
            if ema50_4h_aligned[i] > ema200_4h_aligned[i] and low[i] <= vwap_1d_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + price at/above VWAP + volume
            elif ema50_4h_aligned[i] < ema200_4h_aligned[i] and high[i] >= vwap_1d_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses above VWAP or trend reverses
            if high[i] > vwap_1d_aligned[i] or ema50_4h_aligned[i] <= ema200_4h_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses below VWAP or trend reverses
            if low[i] < vwap_1d_aligned[i] or ema50_4h_aligned[i] >= ema200_4h_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Trend_1dVWAP_MeanReversion_v1"
timeframe = "4h"
leverage = 1.0