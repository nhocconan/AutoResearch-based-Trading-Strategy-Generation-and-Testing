#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d Donchian breakout with 1w trend filter and volume confirmation
    # Donchian(20) breakout captures strong momentum moves
    # 1w EMA50 filter ensures we trade with the higher timeframe trend
    # Volume spike confirms institutional participation
    # Works in bull (breakouts up) and bear (breakouts down) with trend filter
    # Target: 10-25 trades/year per symbol.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Donchian channels (20-period)
    donchian_h = np.full(len(df_1d), np.nan)
    donchian_l = np.full(len(df_1d), np.nan)
    
    for i in range(19, len(df_1d)):
        donchian_h[i] = np.max(high_1d[i-19:i+1])
        donchian_l[i] = np.min(low_1d[i-19:i+1])
    
    # Align 1d Donchian levels to 1d timeframe (no alignment needed for same TF)
    donchian_h_aligned = donchian_h
    donchian_l_aligned = donchian_l
    
    # 1d volume spike filter (current volume > 2.0 * 20-day average)
    vol_ma_20_1d = np.full(len(df_1d), np.nan)
    for i in range(19, len(df_1d)):
        vol_ma_20_1d[i] = np.mean(volume_1d[i-19:i+1])
    volume_spike = volume > 2.0 * vol_ma_20_1d
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA(50) for trend filter
    ema_50_1w = np.full(len(df_1w), np.nan)
    close_1w_series = pd.Series(close_1w)
    ema_50_1w_values = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w[:len(ema_50_1w_values)] = ema_50_1w_values
    
    # Align 1w EMA50 to 1d timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_h_aligned[i]) or np.isnan(donchian_l_aligned[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1w EMA50
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Breakout entries with volume confirmation
        long_entry = (close[i] > donchian_h_aligned[i] and 
                     volume_spike[i] and uptrend)
        short_entry = (close[i] < donchian_l_aligned[i] and 
                      volume_spike[i] and downtrend)
        
        # Exit on opposite Donchian test or volume dropout
        long_exit = close[i] < donchian_l_aligned[i] or (not volume_spike[i])
        short_exit = close[i] > donchian_h_aligned[i] or (not volume_spike[i])
        
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

name = "1d_1w_donchian_breakout_vol_trend_v1"
timeframe = "1d"
leverage = 1.0