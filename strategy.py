#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d Donchian breakout with 1w ATR regime filter and volume confirmation
    # Long: Close breaks above 20-day Donchian upper AND 1w ATR ratio > 1.2 (expanding volatility) AND volume > 1.5x avg
    # Short: Close breaks below 20-day Donchian lower AND 1w ATR ratio > 1.2 AND volume > 1.5x avg
    # Exit: Close crosses Donchian midpoint OR volatility contracts (ATR ratio < 0.8)
    # Using 1d timeframe for low trade frequency, Donchian for structure,
    # 1w ATR ratio for regime filter (avoid choppy markets), volume for confirmation.
    # Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for ATR regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly ATR(14) for regime filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = 0  # First period has no roll
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR calculation with Wilder's smoothing
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: smoothed = (prev * (period-1) + current) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_1w = wilders_smoothing(tr, 14)
    
    # Calculate weekly ATR ratio (current ATR / 50-period mean ATR) for regime filter
    atr_ma_50 = np.full_like(atr_1w, np.nan)
    for i in range(50, len(atr_1w)):
        atr_ma_50[i] = np.mean(atr_1w[i-50:i])
    atr_ratio_1w = np.where(atr_ma_50 > 0, atr_1w / atr_ma_50, 1.0)
    
    # Align weekly ATR ratio to 1d
    atr_ratio_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_ratio_1w)
    
    # Calculate daily Donchian channels (20-period)
    donchian_lookback = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    midpoint = np.full(n, np.nan)
    
    for i in range(donchian_lookback-1, n):
        upper[i] = np.max(high[i-donchian_lookback+1:i+1])
        lower[i] = np.min(low[i-donchian_lookback+1:i+1])
        midpoint[i] = (upper[i] + lower[i]) / 2
    
    # Get daily volume for confirmation (>1.5x 20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(atr_ratio_1w_aligned[i]) or np.isnan(upper[i]) or 
            np.isnan(lower[i]) or np.isnan(midpoint[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: ATR ratio > 1.2 = expanding volatility (trending market)
        volatile_regime = atr_ratio_1w_aligned[i] > 1.2
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Entry logic: Donchian breakout + volatile regime + volume confirmation
        long_entry = (close[i] > upper[i]) and volatile_regime and vol_confirm
        short_entry = (close[i] < lower[i]) and volatile_regime and vol_confirm
        
        # Exit logic: Donchian midpoint cross OR volatility contracts (ATR ratio < 0.8)
        long_exit = (close[i] < midpoint[i]) or (atr_ratio_1w_aligned[i] < 0.8)
        short_exit = (close[i] > midpoint[i]) or (atr_ratio_1w_aligned[i] < 0.8)
        
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

name = "1d_1w_donchian_atr_volume_v1"
timeframe = "1d"
leverage = 1.0