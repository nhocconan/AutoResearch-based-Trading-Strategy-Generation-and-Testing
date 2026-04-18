#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1dTrendVolume
Hypothesis: Breakout of 20-period Donchian channel on 12h with 1d trend and volume confirmation. Works in bull by catching breakouts and in bear by shorting breakdowns. Uses 1d EMA50 for trend filter and 1d volume > 1.5x 20-period average for confirmation. Targets 15-25 trades/year via strict Donchian breakout conditions + 1d filters. Uses volatility-based position sizing (0.25) to manage drawdown.
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
    
    # Get 1d data for trend and volume filters
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 for trend
    close_1d = df_1d['close'].values
    ema_50 = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        ema_50[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50[i] = close_1d[i] * 0.04 + ema_50[i-1] * 0.96
    
    # 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_20 = np.full_like(vol_1d, np.nan)
    if len(vol_1d) >= 20:
        for i in range(20, len(vol_1d)):
            vol_ma_20[i] = np.mean(vol_1d[i-20:i])
    
    # Align 1d indicators to 12h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Donchian channel (20-period) on 12h
    donchian_high = np.full_like(high, np.nan)
    donchian_low = np.full_like(low, np.nan)
    period = 20
    if len(high) >= period:
        for i in range(period-1, len(high)):
            donchian_high[i] = np.max(high[i-period+1:i+1])
            donchian_low[i] = np.min(low[i-period+1:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(period-1, 50)  # Wait for Donchian and EMA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x 20-period 1d average
        vol_confirm = volume[i] > 1.5 * vol_ma_20_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian high + 1d uptrend + volume
            if close[i] > donchian_high[i] and close[i] > ema_50_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + 1d downtrend + volume
            elif close[i] < donchian_low[i] and close[i] < ema_50_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian low or 1d downtrend
            if close[i] < donchian_low[i] or close[i] < ema_50_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian high or 1d uptrend
            if close[i] > donchian_high[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dTrendVolume"
timeframe = "12h"
leverage = 1.0