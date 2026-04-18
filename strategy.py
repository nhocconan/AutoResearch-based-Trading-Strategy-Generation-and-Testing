#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrendVolume
Hypothesis: Trade 4h Donchian breakouts with 1d trend and volume confirmation.
Long when price breaks above 4h Donchian high (20) + 1d close > 1d EMA20 + volume > 1.5x 24-bar avg.
Short when price breaks below 4h Donchian low (20) + 1d close < 1d EMA20 + volume > 1.5x 24-bar avg.
Exit on opposite Donchian breakout or trend reversal.
Targets 20-40 trades/year via Donchian breakouts + 1d trend filter.
Works in bull/bear by following 1d trend direction.
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA20 for trend
    close_1d = df_1d['close'].values
    ema_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 20:
        ema_1d[0] = close_1d[0]
        alpha = 2 / (20 + 1)
        for i in range(1, len(close_1d)):
            ema_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_1d[i-1]
    
    # Align 1d EMA to 4h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Get 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # 4h Donchian channels (20-period)
    donch_high = np.full_like(close_4h, np.nan)
    donch_low = np.full_like(close_4h, np.nan)
    
    if len(close_4h) >= 20:
        for i in range(20, len(close_4h)):
            donch_high[i] = np.max(high_4h[i-20:i])
            donch_low[i] = np.min(low_4h[i-20:i])
    
    # Align Donchian channels to 4h timeframe (same as input, but using alignment for consistency)
    donch_high_aligned = align_htf_to_ltf(prices, df_4h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_4h, donch_low)
    
    # Volume confirmation: volume > 1.5x 24-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 24
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20, vol_period)  # Donchian needs 20, vol MA needs 24
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Donchian breakout + 1d uptrend + volume
            if close[i] > donch_high_aligned[i] and close_4h[-1] > ema_1d_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakdown + 1d downtrend + volume
            elif close[i] < donch_low_aligned[i] and close_4h[-1] < ema_1d_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Donchian breakdown or 1d downtrend
            if close[i] < donch_low_aligned[i] or close_4h[-1] < ema_1d_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Donchian breakout or 1d uptrend
            if close[i] > donch_high_aligned[i] or close_4h[-1] > ema_1d_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dTrendVolume"
timeframe = "4h"
leverage = 1.0