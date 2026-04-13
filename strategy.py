#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout + 12h HMA(21) trend filter + volume confirmation
    # Long: price > Donchian upper(20) AND HMA(21) rising AND volume > 1.5 * volume_ma(20)
    # Short: price < Donchian lower(20) AND HMA(21) falling AND volume > 1.5 * volume_ma(20)
    # Exit: opposite Donchian break OR HMA trend reversal
    # Using discrete sizing (0.25) to minimize fee churn and control drawdown
    # Target: 20-50 trades/year (~80-200 over 4 years) to avoid overtrading
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for HMA trend filter (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h HMA(21)
    def hull_moving_average(arr, period):
        """Hull Moving Average: WMA(2*WMA(n/2) - WMA(n), sqrt(n))"""
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        half = period // 2
        sqrt_period = int(np.sqrt(period))
        
        def wma(data, window):
            if len(data) < window:
                return np.full_like(data, np.nan)
            weights = np.arange(1, window + 1)
            return np.convolve(data, weights, mode='valid') / weights.sum()
        
        wma_half = wma(arr, half)
        wma_full = wma(arr, period)
        if len(wma_half) < 1 or len(wma_full) < 1:
            return np.full_like(arr, np.nan)
        # Align arrays: wma_half starts at index (period - half)
        # wma_full starts at index (period - 1)
        diff = 2 * wma_half[-len(wma_full):] - wma_full
        if len(diff) < sqrt_period:
            return np.full_like(arr, np.nan)
        hma = wma(diff, sqrt_period)
        # Pad beginning with NaN
        hma_full = np.full_like(arr, np.nan)
        hma_full[period - 1:] = hma
        return hma_full
    
    close_12h = df_12h['close'].values
    hma_21_12h = hull_moving_average(close_12h, 21)
    hma_21_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_21_12h)
    
    # Calculate 4h Donchian channels (20-period)
    def donchian_channels(high, low, period):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(low, np.nan)
        for i in range(period - 1, len(high)):
            upper[i] = np.max(high[i - period + 1:i + 1])
            lower[i] = np.min(low[i - period + 1:i + 1])
        return upper, lower
    
    upper_20, lower_20 = donchian_channels(high, low, 20)
    
    # Calculate 4h volume MA(20) for confirmation
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(hma_21_12h_aligned[i]) or np.isnan(upper_20[i]) or 
            np.isnan(lower_20[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: HMA rising/falling
        hma_rising = hma_21_12h_aligned[i] > hma_21_12h_aligned[i-1]
        hma_falling = hma_21_12h_aligned[i] < hma_21_12h_aligned[i-1]
        
        # Volume confirmation: current volume > 1.5 * 20-period MA
        volume_confirm = volume[i] > 1.5 * volume_ma_20[i]
        
        # Donchian breakout signals
        long_break = close[i] > upper_20[i]
        short_break = close[i] < lower_20[i]
        
        # Entry logic: Donchian break + trend filter + volume confirmation
        long_entry = long_break and hma_rising and volume_confirm
        short_entry = short_break and hma_falling and volume_confirm
        
        # Exit logic: opposite Donchian break OR HMA trend reversal
        long_exit = short_break or not hma_rising
        short_exit = long_break or not hma_falling
        
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

name = "4h_12h_donchian_hma_volume_v1"
timeframe = "4h"
leverage = 1.0