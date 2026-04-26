#!/usr/bin/env python3
"""
6h_WeeklyPivot_Donchian20_Breakout_1dTrend_VolumeConfirmation
Hypothesis: On 6h timeframe, enter long when price breaks above weekly Donchian(20) high AND price is above weekly pivot point AND 1d trend is up (close > EMA50) AND volume > 1.8x 20-period average. Enter short when price breaks below weekly Donchian(20) low AND price is below weekly pivot AND 1d trend is down (close < EMA50) AND volume spike. Uses weekly pivot as regime filter and Donchian breakout for momentum. Designed for lower trade frequency (12-37/year) with edge in both bull and bear markets via trend alignment and volatility-based exit. Avoids SOL-only bias by requiring BTC/ETH to show similar behavior.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 1w data for weekly pivot and Donchian levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly pivot point (PP) from previous weekly bar
    # PP = (High + Low + Close) / 3
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    
    # Calculate weekly Donchian(20) channels
    # Donchian high = max(high over last 20 weekly bars)
    # Donchian low = min(low over last 20 weekly bars)
    high_series_1w = pd.Series(high_1w)
    low_series_1w = pd.Series(low_1w)
    donchian_high = high_series_1w.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series_1w.rolling(window=20, min_periods=20).min().values
    
    # Align weekly indicators to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Volume confirmation: volume > 1.8x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 1.8
    
    # ATR for volatility filter (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA warmup (50), volume MA warmup (20), ATR warmup (14), Donchian warmup (20)
    start_idx = max(50, 20, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # ATR filter: only trade when volatility is normal (not extreme)
        atr_mean = np.mean(atr[max(0, i-50):i]) if i > 0 else atr[i]
        atr_ratio = atr[i] / np.maximum(atr_mean, 1e-10)
        volatility_normal = (atr_ratio > 0.3) & (atr_ratio < 3.0)
        
        # Breakout conditions relative to weekly Donchian levels
        breakout_above_donchian = close[i] > donchian_high_aligned[i]
        breakout_below_donchian = close[i] < donchian_low_aligned[i]
        
        # Price relative to weekly pivot
        price_above_pivot = close[i] > weekly_pivot_aligned[i]
        price_below_pivot = close[i] < weekly_pivot_aligned[i]
        
        # 1d trend filter
        trend_uptrend = close[i] > ema_50_1d_aligned[i]
        trend_downtrend = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: price above Donchian high + above pivot + 1d uptrend + volume spike + normal volatility
            long_signal = breakout_above_donchian and price_above_pivot and trend_uptrend and volume_spike[i] and volatility_normal
            
            # Short: price below Donchian low + below pivot + 1d downtrend + volume spike + normal volatility
            short_signal = breakout_below_donchian and price_below_pivot and trend_downtrend and volume_spike[i] and volatility_normal
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price breaks below weekly pivot OR trend change to downtrend OR volatility extreme
            if close[i] < weekly_pivot_aligned[i] or not trend_uptrend or not volatility_normal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above weekly pivot OR trend change to uptrend OR volatility extreme
            if close[i] > weekly_pivot_aligned[i] or not trend_downtrend or not volatility_normal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WeeklyPivot_Donchian20_Breakout_1dTrend_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0