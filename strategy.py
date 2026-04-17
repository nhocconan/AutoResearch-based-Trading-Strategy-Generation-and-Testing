#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_Volume_Trend_v1
Donchian channel breakout with volume confirmation and trend filter on 4h timeframe.
Uses 20-period Donchian bands for breakouts, volume spike for confirmation, and
1-day EMA trend filter to align with higher timeframe direction.
Designed to capture strong momentum moves while avoiding choppy markets.
Target: 75-200 total trades over 4 years (19-50/year).
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
    
    # === 4h Donchian Channel (20-period) ===
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(n):
        if i >= 19:
            donchian_high[i] = np.max(high[i-19:i+1])
            donchian_low[i] = np.min(low[i-19:i+1])
        else:
            donchian_high[i] = np.nan
            donchian_low[i] = np.nan
    
    # === 4h Volume Confirmation (20-period average) ===
    vol_ma_20 = np.full(n, np.nan)
    for i in range(n):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20[i] = np.nan
    vol_confirm = volume > vol_ma_20 * 1.5  # volume spike: 1.5x average
    
    # === 1d EMA Trend Filter (34-period) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = np.full_like(close_1d, np.nan)
    for i in range(len(close_1d)):
        if i >= 33:
            if i == 33:
                ema_34_1d[i] = np.mean(close_1d[0:34])
            else:
                ema_34_1d[i] = (close_1d[i] * 2 / 35) + (ema_34_1d[i-1] * 33 / 35)
        else:
            ema_34_1d[i] = np.nan
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(vol_confirm[i]) or 
            np.isnan(ema_34_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: break above upper Donchian band + volume confirmation + price above 1d EMA
            if (close[i] > donchian_high[i] and 
                vol_confirm[i] and 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: break below lower Donchian band + volume confirmation + price below 1d EMA
            elif (close[i] < donchian_low[i] and 
                  vol_confirm[i] and 
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price crosses below lower Donchian band OR volume drops
            if (close[i] < donchian_low[i] or 
                not vol_confirm[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above upper Donchian band OR volume drops
            if (close[i] > donchian_high[i] or 
                not vol_confirm[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_Volume_Trend_v1"
timeframe = "4h"
leverage = 1.0