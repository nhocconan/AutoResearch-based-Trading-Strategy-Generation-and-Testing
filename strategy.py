#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_camarilla_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla levels and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly ATR (14-period) for volatility filter
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate ATR ratio (current ATR / 20-period ATR mean) for volatility scaling
    atr_ma_20 = pd.Series(atr_1w).rolling(window=20, min_periods=20).mean().values
    atr_ratio = atr_1w / atr_ma_20
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1w, atr_ratio)
    
    # Calculate weekly Camarilla levels (using previous week's data)
    camarilla_high = np.full(len(close_1w), np.nan)
    camarilla_low = np.full(len(close_1w), np.nan)
    
    for i in range(1, len(close_1w)):
        H = high_1w[i-1]
        L = low_1w[i-1]
        C = close_1w[i-1]
        camarilla_high[i] = C + ((H - L) * 1.1 / 2)
        camarilla_low[i] = C - ((H - L) * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe
    camarilla_high_aligned = align_htf_to_ltf(prices, df_1w, camarilla_high)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_1w, camarilla_low)
    
    # Calculate weekly pivot point (for exit)
    pivot_point = np.full(len(close_1w), np.nan)
    for i in range(1, len(close_1w)):
        H = high_1w[i-1]
        L = low_1w[i-1]
        C = close_1w[i-1]
        pivot_point[i] = (H + L + C) / 3
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot_point)
    
    # Volume filter: current volume > 20-period average (on 6h data)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma
    
    # Determine weekly trend using close vs 20-week SMA
    sma_20 = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    weekly_trend = np.where(close_1w > sma_20, 1, -1)  # 1=uptrend, -1=downtrend
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend)
    
    # Base position size with volatility scaling (inverse volatility)
    # Higher volatility = smaller position, capped at 0.30
    vol_scaling = np.clip(1.0 / (atr_ratio_aligned + 0.001), 0.5, 1.5)
    base_size = 0.25
    position_size = base_size * vol_scaling
    position_size = np.clip(position_size, 0.10, 0.30)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(camarilla_high_aligned[i]) or np.isnan(camarilla_low_aligned[i]) or 
            np.isnan(volume_ok[i]) or np.isnan(position_size[i]) or np.isnan(weekly_trend_aligned[i])):
            signals[i] = 0.0 if position == 0 else (position_size[i] if position == 1 else -position_size[i])
            continue
        
        # Breakout conditions with volume confirmation and trend filter
        breakout_up = close[i] > camarilla_high_aligned[i]
        breakout_down = close[i] < camarilla_low_aligned[i]
        vol_ok = volume_ok[i]
        trend_up = weekly_trend_aligned[i] == 1
        trend_down = weekly_trend_aligned[i] == -1
        
        # Entry signals: only trade in direction of weekly trend
        long_signal = breakout_up and vol_ok and trend_up
        short_signal = breakout_down and vol_ok and trend_down
        
        # Exit when price returns to the weekly pivot
        exit_long = close[i] < pivot_aligned[i]
        exit_short = close[i] > pivot_aligned[i]
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = position_size[i]
        elif short_signal and position != -1:
            position = -1
            signals[i] = -position_size[i]
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position with dynamic sizing
            if position == 1:
                signals[i] = position_size[i]
            elif position == -1:
                signals[i] = -position_size[i]
            else:
                signals[i] = 0.0
    
    return signals