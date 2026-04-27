#!/usr/bin/env python3
"""
4h_DonchianBreakout_12hEMA50_Volume_ReversalExit
Hypothesis: Donchian(20) breakout on 4h with 12h EMA50 trend filter and volume confirmation.
Exit when price closes below/above 12h EMA50 (reversal signal) to avoid whipsaws.
Designed to work in both bull and bear markets by following trend with volatility breakout.
Target: 20-50 trades/year to minimize fee drag.
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
    
    # Get 4h data for Donchian calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate Donchian(20) on 4h high/low
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    donchian_high = np.full(len(high_4h), np.nan)
    donchian_low = np.full(len(low_4h), np.nan)
    
    period = 20
    for i in range(period - 1, len(high_4h)):
        donchian_high[i] = np.max(high_4h[i - period + 1:i + 1])
        donchian_low[i] = np.min(low_4h[i - period + 1:i + 1])
    
    # Align Donchian levels to 4h timeframe
    dh_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    dl_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 12h close
    close_12h = df_12h['close'].values
    ema_period = 50
    ema_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= ema_period:
        ema_12h[ema_period - 1] = np.mean(close_12h[:ema_period])
        multiplier = 2 / (ema_period + 1)
        for i in range(ema_period, len(close_12h)):
            ema_12h[i] = (close_12h[i] * multiplier) + (ema_12h[i - 1] * (1 - multiplier))
    
    # Align 12h EMA to 4h timeframe
    ema_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume confirmation
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period, n):
        vol_ma[i] = np.mean(volume[i - vol_ma_period:i])
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup: need all indicators
    start_idx = max(period, ema_period, vol_ma_period)
    
    for i in range(start_idx, n):
        if (np.isnan(dh_aligned[i]) or
            np.isnan(dl_aligned[i]) or
            np.isnan(ema_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Trend filter: price above/below 12h EMA50
        uptrend = price > ema_aligned[i]
        downtrend = price < ema_aligned[i]
        
        # Volume confirmation: > 1.5x average volume
        volume_confirmation = vol_ratio > 1.5
        
        if position == 0:
            # Long: Donchian breakout above upper band
            if uptrend and volume_confirmation and price > dh_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakdown below lower band
            elif downtrend and volume_confirmation and price < dl_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price closes below 12h EMA50 (trend reversal)
            if price < ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # Maintain position
        elif position == -1:
            # Short exit: price closes above 12h EMA50 (trend reversal)
            if price > ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # Maintain position
    
    return signals

name = "4h_DonchianBreakout_12hEMA50_Volume_ReversalExit"
timeframe = "4h"
leverage = 1.0