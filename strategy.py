#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Donchian(20) breakout with 1d volume confirmation and 1w trend filter
    # Weekly EMA50 provides multi-week trend bias to avoid counter-trend trades
    # Daily volume > 2.0x 20-period average confirms institutional participation
    # Target: 12-25 trades/year (50-100 total over 4 years) for low fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    vol_1d = df_1d['volume'].values
    
    # Get 1w data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 for trend filter
    ema50_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 50:
        ema50_1w[49] = np.mean(close_1w[:50])  # SMA for first value
        alpha = 2.0 / (50 + 1)
        for i in range(50, len(close_1w)):
            ema50_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema50_1w[i-1]
    
    # Get 6h Donchian(20) for breakout
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Get 6h volume for confirmation (>2.0x 20-period average)
    vol_ma_6h = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_6h[i] = np.mean(volume[i-20:i])
    volume_spike_6h = volume > (2.0 * vol_ma_6h)
    
    # Align all indicators to LTF (6h)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_6h)  # Reuse 6h vol MA for alignment structure
    volume_spike_1d = align_htf_to_ltf(prices, df_1d, vol_1d > (2.0 * np.full(len(vol_1d), np.nan)))  # Will compute properly below
    
    # Recompute 1d volume spike properly
    vol_ma_1d = np.full(len(vol_1d), np.nan)
    for i in range(20, len(vol_1d)):
        vol_ma_1d[i] = np.mean(vol_1d[i-20:i])
    volume_spike_1d_raw = vol_1d > (2.0 * vol_ma_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d_raw)
    
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    donchian_high_aligned = donchian_high  # Already LTF
    donchian_low_aligned = donchian_low    # Already LTF
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(volume_spike_6h[i]) or np.isnan(volume_spike_1d_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        long_breakout = close[i] > donchian_high_aligned[i]
        short_breakout = close[i] < donchian_low_aligned[i]
        
        # Trend filter: price above/below weekly EMA50
        bullish_trend = close[i] > ema50_1w_aligned[i]
        bearish_trend = close[i] < ema50_1w_aligned[i]
        
        # Volume confirmation: both 6h and 1d spikes required
        volume_confirmed = volume_spike_6h[i] and volume_spike_1d_aligned[i]
        
        # Entry logic: Breakout + trend alignment + volume confirmation
        long_entry = long_breakout and bullish_trend and volume_confirmed
        short_entry = short_breakout and bearish_trend and volume_confirmed
        
        # Exit logic: opposite breakout or loss of trend
        long_exit = short_breakout or not bullish_trend
        short_exit = long_breakout or not bearish_trend
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1w_1d_donchian_breakout_volume_trend_filter_v1"
timeframe = "6h"
leverage = 1.0