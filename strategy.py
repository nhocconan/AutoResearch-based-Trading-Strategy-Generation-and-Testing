#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Elder Ray Index with 1w trend filter and volume confirmation.
    # Long when 1w close > 1w SMA50 (bullish trend) AND Bull Power > 0 AND 6h volume > 1.5x 20-period MA.
    # Short when 1w close < 1w SMA50 (bearish trend) AND Bear Power < 0 AND 6h volume > 1.5x 20-period MA.
    # Exit when power crosses zero (mean reversion).
    # Uses Elder Ray for momentum, weekly SMA for trend filter, volume for confirmation.
    # Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    sma_50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    # Get daily data for Elder Ray calculation (13-period EMA)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 13-period EMA for Elder Ray
    ema_13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high_1d - ema_13
    bear_power = low_1d - ema_13
    
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Get 6h data for volume confirmation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    volume_6h = df_6h['volume'].values
    vol_ma_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(sma_50_1w_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or np.isnan(vol_ma_6h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x 20-period average
        volume_6h_aligned = align_htf_to_ltf(prices, df_6h, volume_6h)
        volume_spike = volume_6h_aligned[i] > 1.5 * vol_ma_6h_aligned[i]
        
        # Trend filter: weekly close relative to weekly SMA50
        close_1w_aligned = align_htf_to_ltf(prices, df_1w, df_1w['close'].values)
        weekly_uptrend = close_1w_aligned[i] > sma_50_1w_aligned[i]
        weekly_downtrend = close_1w_aligned[i] < sma_50_1w_aligned[i]
        
        # Elder Ray conditions
        bullish_momentum = bull_power_aligned[i] > 0
        bearish_momentum = bear_power_aligned[i] < 0
        exit_signal = (bull_power_aligned[i] * bull_power_aligned[i-1] <= 0) or \
                      (bear_power_aligned[i] * bear_power_aligned[i-1] <= 0)
        
        # Entry conditions
        if weekly_uptrend and bullish_momentum and volume_spike and position != 1:
            position = 1
            signals[i] = position_size
        elif weekly_downtrend and bearish_momentum and volume_spike and position != -1:
            position = -1
            signals[i] = -position_size
        # Exit conditions
        elif exit_signal and position != 0:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1w_1d_elder_ray_trend_volume_v1"
timeframe = "6h"
leverage = 1.0