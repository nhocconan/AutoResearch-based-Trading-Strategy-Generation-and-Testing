#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with 1d EMA trend filter and volume confirmation
# Uses Donchian(20) breakouts in direction of 1d EMA(50) trend, with volume > 1.5x 20-period average.
# Works in both bull and bear markets by only taking breakouts aligned with higher timeframe trend.
# Target: 20-40 trades/year to minimize fee decay while capturing strong trending moves.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d
    close_1d = df_1d['close'].values
    ema_period = 50
    ema_1d = np.full_like(close_1d, np.nan)
    
    if len(close_1d) >= ema_period:
        # Simple average for first value
        ema_1d[ema_period-1] = np.mean(close_1d[:ema_period])
        # EMA for subsequent values
        alpha = 2.0 / (ema_period + 1.0)
        for i in range(ema_period, len(close_1d)):
            ema_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_1d[i-1]
    
    # Donchian channel on 4h (20-period)
    dc_period = 20
    upper_channel = np.full(n, np.nan)
    lower_channel = np.full(n, np.nan)
    
    for i in range(dc_period, n):
        upper_channel[i] = np.max(high[i-dc_period:i])
        lower_channel[i] = np.min(low[i-dc_period:i])
    
    # 20-period average volume for spike detection
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    # Align HTF indicators to 4h
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup period
    start_idx = max(dc_period, vol_period) + ema_period
    
    for i in range(start_idx, n):
        if (np.isnan(ema_1d_aligned[i]) or 
            np.isnan(upper_channel[i]) or 
            np.isnan(lower_channel[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Conditions:
        # 1. EMA trend filter: price above EMA(50) for long, below for short
        # 2. Volume confirmation: > 1.5x average volume
        # 3. Donchian breakout: price breaks above upper channel (long) or below lower channel (short)
        trend_long = price > ema_1d_aligned[i]
        trend_short = price < ema_1d_aligned[i]
        volume_confirmation = vol_ratio > 1.5
        breakout_up = price > upper_channel[i]
        breakout_down = price < lower_channel[i]
        
        if position == 0:
            # Long: breakout above upper channel with uptrend and volume
            if trend_long and volume_confirmation and breakout_up:
                signals[i] = size
                position = 1
            # Short: breakout below lower channel with downtrend and volume
            elif trend_short and volume_confirmation and breakout_down:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price returns to lower Donchian channel
            if price < lower_channel[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price returns to upper Donchian channel
            if price > upper_channel[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian_EMATrend_Volume"
timeframe = "4h"
leverage = 1.0