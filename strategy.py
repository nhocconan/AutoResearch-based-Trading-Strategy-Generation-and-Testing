#!/usr/bin/env python3
"""
4h_Donchian_Breakout_20_Trend_Volume
Hypothesis: Price breaks Donchian(20) high/low with trend confirmation from 1d EMA200 and volume > 1.5x 20-day average. Donchian breakouts capture momentum in trending markets; EMA200 ensures alignment with higher timeframe trend; volume confirmation filters false breakouts. Designed for low trade frequency (<30/year) to minimize fee drag and work in both bull and bear markets.
"""

name = "4h_Donchian_Breakout_20_Trend_Volume"
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
    
    # Donchian(20) channels - calculated on 4h data
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback-1, n):
        highest_high[i] = np.max(high[i-lookback+1:i+1])
        lowest_low[i] = np.min(low[i-lookback+1:i+1])
    
    # 1d EMA200 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_200_1d = np.full(len(close_1d), np.nan)
    
    if len(close_1d) >= 200:
        # Calculate EMA200 with proper initialization
        ema_200_1d[199] = np.mean(close_1d[:200])
        alpha = 2 / (200 + 1)
        for i in range(200, len(close_1d)):
            ema_200_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_200_1d[i-1]
    
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # 1d volume SMA20 for volume confirmation
    volume_1d = df_1d['volume'].values
    vol_sma_20_1d = np.full(len(volume_1d), np.nan)
    
    if len(volume_1d) >= 20:
        vol_sma_20_1d[19] = np.mean(volume_1d[:20])
        for i in range(20, len(volume_1d)):
            vol_sma_20_1d[i] = (vol_sma_20_1d[i-1] * 19 + volume_1d[i]) / 20
    
    vol_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, 20)  # warmup for EMA200 and Donchian
    
    for i in range(start_idx, n):
        if np.isnan(ema_200_1d_aligned[i]) or np.isnan(vol_sma_20_1d_aligned[i]) or \
           np.isnan(highest_high[i]) or np.isnan(lowest_low[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x average 1d volume (scaled to 4h)
        vol_4h_approx = vol_sma_20_1d_aligned[i] / 6.0  # 24h/4h = 6
        volume_confirm = volume[i] > 1.5 * vol_4h_approx
        
        if position == 0:
            # Long: Break above Donchian high with uptrend and volume
            if close[i] > highest_high[i] and close[i] > ema_200_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low with downtrend and volume
            elif close[i] < lowest_low[i] and close[i] < ema_200_1d_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close below Donchian low (trend reversal)
            if close[i] < lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above Donchian high (trend reversal)
            if close[i] > highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals