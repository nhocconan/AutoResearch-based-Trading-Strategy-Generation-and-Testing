#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d volume confirmation and 1w trend filter.
# Uses Donchian channel breakouts as primary signal, confirmed by volume spikes
# and filtered by weekly trend direction. Designed for low trade frequency
# (target: 20-50 trades/year) to minimize fee drag while capturing strong trends.
# Works in both bull and bear markets by following the weekly trend.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for volume confirmation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Volume moving average (20-period) for confirmation
    vol_ma_1d = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-20:i])
    
    # Weekly trend filter (EMA 50 on 1w data)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 50:
        multiplier = 2 / (50 + 1)
        ema50_1w[49] = np.mean(close_1w[:50])
        for i in range(50, len(close_1w)):
            ema50_1w[i] = (close_1w[i] - ema50_1w[i-1]) * multiplier + ema50_1w[i-1]
    
    # Align 1d volume MA and 1w EMA to 4h timeframe
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Donchian channel (20-period) on 4h
    upper_channel = np.full(n, np.nan)
    lower_channel = np.full(n, np.nan)
    
    for i in range(20, n):
        upper_channel[i] = np.max(high[i-20:i])
        lower_channel[i] = np.min(low[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(ema50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_1d_aligned[i]
        ema_trend = ema50_1w_aligned[i]
        
        # Volume confirmation: current volume > 2.0x average daily volume
        volume_confirm = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long: Price breaks above upper channel + above weekly EMA + volume confirmation
            if (price > upper_channel[i] and 
                price > ema_trend and 
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: Price breaks below lower channel + below weekly EMA + volume confirmation
            elif (price < lower_channel[i] and 
                  price < ema_trend and 
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Price breaks below lower channel or below weekly EMA
            if (price < lower_channel[i] or price < ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Price breaks above upper channel or above weekly EMA
            if (price > upper_channel[i] or price > ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_1w_Donchian_Volume_Trend"
timeframe = "4h"
leverage = 1.0