#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Donchian(20) breakout + 1d volume spike + 1w trend filter
    # Long: price breaks above 6h Donchian high AND 1d volume > 1.5 * 20-period average AND price > 1w EMA50
    # Short: price breaks below 6h Donchian low AND 1d volume > 1.5 * 20-period average AND price < 1w EMA50
    # Exit: price reverts to 6h Donchian midpoint
    # Using 1d for volume confirmation (avoids look-ahead) and 1w for trend filter
    # Discrete position sizing (0.25) to minimize fee churn
    # Target: 12-37 trades/year (~50-150 over 4 years) to stay within fee drag limits
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Donchian channels (call ONCE before loop)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 30:
        return np.zeros(n)
    
    # Get 1d data for volume spike (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 1w data for trend filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 6h Donchian channels (20-period)
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Donchian high: rolling max of high
    donchian_high = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    # Donchian low: rolling min of low
    donchian_low = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    # Donchian midpoint: average of high and low
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Align 6h Donchian to 6h timeframe (no additional delay needed for price channels)
    donchian_high_aligned = align_htf_to_ltf(prices, df_6h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_6h, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_6h, donchian_mid)
    
    # Calculate 1d volume spike filter: volume > 1.5 * 20-period average
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (1.5 * vol_ma_20)
    
    # Align 1d volume spike to 6h (wait for completed 1d bar)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 6h (wait for completed 1w bar)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(volume_spike_aligned[i]) or 
            np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1d volume spike
        vol_confirmed = volume_spike_aligned[i] > 0.5  # boolean as float
        
        # Trend filter: price relative to 1w EMA50
        uptrend = close[i] > ema_50_aligned[i]
        downtrend = close[i] < ema_50_aligned[i]
        
        # Breakout conditions
        long_breakout = close[i] > donchian_high_aligned[i]
        short_breakout = close[i] < donchian_low_aligned[i]
        
        # Entry logic: Donchian breakout + volume spike + trend filter
        long_entry = long_breakout and vol_confirmed and uptrend
        short_entry = short_breakout and vol_confirmed and downtrend
        
        # Exit logic: price reverts to midpoint
        long_exit = close[i] < donchian_mid_aligned[i]
        short_exit = close[i] > donchian_mid_aligned[i]
        
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

name = "6h_1d_1w_donchian_volume_trend_v1"
timeframe = "6h"
leverage = 1.0