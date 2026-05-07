#!/usr/bin/env python3

# 4h_Donchian20_Breakout_1dTrend_VolumeConfirm
# Hypothesis: 4-hour Donchian(20) breakouts filtered by 1-day EMA(34) trend direction and volume confirmation (current volume > 1.5x 20-period average).
# The strategy aims to capture strong momentum moves aligned with the daily trend while avoiding choppy markets.
# Target: 20-50 total trades over 4 years (5-12/year) to minimize fee drag and improve generalization.

name = "4h_Donchian20_Breakout_1dTrend_VolumeConfirm"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period) using previous candle
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 34:
        ema_34_1d[33] = np.mean(close_1d[:34])  # Simple average for first value
        for i in range(34, len(close_1d)):
            ema_34_1d[i] = (close_1d[i] * 2 / 35) + (ema_34_1d[i-1] * 33 / 35)
    
    # Calculate 20-period volume average for confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    # Align 1d EMA to 4h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 6  # Minimum 1 day between trades (6 * 4h = 24h)
    
    start_idx = max(20, 34)  # Warmup for Donchian and EMA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Donchian breakout above upper band, daily uptrend, volume confirmation
            if (close[i] > donchian_high[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume[i] > 1.5 * vol_ma_20[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Donchian breakdown below lower band, daily downtrend, volume confirmation
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume[i] > 1.5 * vol_ma_20[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit long: price crosses below Donchian lower band
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Donchian upper band
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals