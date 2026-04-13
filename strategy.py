#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d KAMA trend filter with 1w Donchian(20) breakout + volume confirmation
    # Long: price > 1w Donchian high AND KAMA rising AND volume > 1.5x 20-period average
    # Short: price < 1w Donchian low AND KAMA falling AND volume > 1.5x 20-period average
    # Exit: opposite Donchian breakout OR KAMA flips direction
    # Using 1d timeframe for low trade frequency (target 7-25/year), 1w Donchian for major structure,
    # and KAMA for adaptive trend filtering. Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Donchian(20) breakout levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 1w Donchian(20) with min_periods
    donchian_high_1w = np.full(len(high_1w), np.nan)
    donchian_low_1w = np.full(len(low_1w), np.nan)
    
    for i in range(20, len(high_1w)):
        donchian_high_1w[i] = np.max(high_1w[i-20:i])
        donchian_low_1w[i] = np.min(low_1w[i-20:i])
    
    # Align 1w Donchian levels to 1d
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_1w)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_1w)
    
    # Calculate KAMA on 1d close prices
    def calculate_kama(close_vals, er_len=10, fast_ema=2, slow_ema=30):
        n = len(close_vals)
        if n < er_len:
            return np.full(n, np.nan)
        
        # Efficiency Ratio
        change = np.abs(np.diff(close_vals, n=er_len))
        volatility = np.sum(np.abs(np.diff(close_vals)), axis=1)
        er = np.full(n, np.nan)
        er[er_len:] = change[er_len-1:] / np.maximum(volatility[er_len-1:], 1e-10)
        
        # Smoothing Constants
        sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
        
        # KAMA calculation
        kama = np.full(n, np.nan)
        kama[er_len-1] = close_vals[er_len-1]
        for i in range(er_len, n):
            kama[i] = kama[i-1] + sc[i] * (close_vals[i] - kama[i-1])
        return kama
    
    kama_1d = calculate_kama(close, er_len=10, fast_ema=2, slow_ema=30)
    
    # Get 1d volume for confirmation (>1.5x 20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(kama_1d[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        long_breakout = close[i] > donchian_high_aligned[i]
        short_breakout = close[i] < donchian_low_aligned[i]
        
        # KAMA trend conditions (rising/falling)
        kama_rising = kama_1d[i] > kama_1d[i-1]
        kama_falling = kama_1d[i] < kama_1d[i-1]
        
        # Entry logic: Breakout + KAMA alignment + volume confirmation
        long_entry = long_breakout and kama_rising and volume_spike[i]
        short_entry = short_breakout and kama_falling and volume_spike[i]
        
        # Exit logic: opposite breakout or KAMA flips direction
        long_exit = short_breakout or not kama_rising
        short_exit = long_breakout or not kama_falling
        
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

name = "1d_1w_kama_donchian_breakout_volume_v1"
timeframe = "1d"
leverage = 1.0