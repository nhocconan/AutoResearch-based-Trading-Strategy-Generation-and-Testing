#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1h strategy using 4h Donchian breakout with 1d EMA200 trend filter and session filter (08-20 UTC)
    # Long: price > 4h Donchian(20) high + price > 1d EMA200 + volume > 1.5x 20-period average + session 08-20 UTC
    # Short: price < 4h Donchian(20) low + price < 1d EMA200 + volume > 1.5x 20-period average + session 08-20 UTC
    # Exit: opposite Donchian breakout OR price crosses 1d EMA200
    # Target: 15-35 trades/year (60-140 total) for low fee drag and strong edge in both bull and bear
    # Uses 4h/1d for signal direction, 1h only for entry timing precision
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Donchian(20)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h Donchian(20)
    donchian_high_4h = np.full(len(high_4h), np.nan)
    donchian_low_4h = np.full(len(low_4h), np.nan)
    
    for i in range(20, len(high_4h)):
        donchian_high_4h[i] = np.max(high_4h[i-20:i])
        donchian_low_4h[i] = np.min(low_4h[i-20:i])
    
    # Align 4h Donchian to 1h
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_4h)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_4h)
    
    # Get 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA200
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 200:
        ema_1d[199] = np.mean(close_1d[:200])  # SMA200 as seed
        multiplier = 2 / (200 + 1)
        for i in range(200, len(close_1d)):
            ema_1d[i] = (close_1d[i] * multiplier) + (ema_1d[i-1] * (1 - multiplier))
    
    # Align 1d EMA200 to 1h
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Get 4h volume for confirmation (>1.5x 20-period average)
    vol_4h = df_4h['volume'].values
    vol_ma_4h = np.full(len(vol_4h), np.nan)
    for i in range(20, len(vol_4h)):
        vol_ma_4h[i] = np.mean(vol_4h[i-20:i])
    volume_spike_4h = vol_4h > (1.5 * vol_ma_4h)
    volume_spike_aligned = align_htf_to_ltf(prices, df_4h, volume_spike_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        long_breakout = close[i] > donchian_high_aligned[i]
        short_breakout = close[i] < donchian_low_aligned[i]
        
        # Trend filter from 1d EMA200
        bullish_trend = close[i] > ema_1d_aligned[i]
        bearish_trend = close[i] < ema_1d_aligned[i]
        
        # Entry logic: Breakout + trend alignment + volume confirmation
        long_entry = long_breakout and bullish_trend and volume_spike_aligned[i]
        short_entry = short_breakout and bearish_trend and volume_spike_aligned[i]
        
        # Exit logic: opposite breakout or trend reversal
        long_exit = short_breakout or (close[i] < ema_1d_aligned[i])
        short_exit = long_breakout or (close[i] > ema_1d_aligned[i])
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_4h_1d_donchian_breakout_ema200_volume_session_v1"
timeframe = "1h"
leverage = 1.0