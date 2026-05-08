#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian channel breakout with 1w trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high AND weekly trend is up AND volume > 1.5x average.
# Short when price breaks below Donchian(20) low AND weekly trend is down AND volume > 1.5x average.
# Uses 1w EMA(20) for trend direction. Includes exit when price crosses Donchian midline or trend reverses.
# Designed to work in both bull and bear markets by following the weekly trend.
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag.

name = "1d_Donchian_20_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # 1w EMA(20) for trend
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    trend_1w_up = ema_20_1w > np.roll(ema_20_1w, 1)
    trend_1w_up = np.where(np.isnan(trend_1w_up), False, trend_1w_up)
    
    # Align 1w trend to 1d
    trend_1w_up_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_up.astype(float))
    
    # Donchian channel (20-period) on 1d data
    donchian_high = np.zeros_like(high_1d)
    donchian_low = np.zeros_like(low_1d)
    donchian_mid = np.zeros_like(close_1d)
    
    for i in range(len(close_1d)):
        if i < 19:
            donchian_high[i] = np.nan
            donchian_low[i] = np.nan
            donchian_mid[i] = np.nan
        else:
            donchian_high[i] = np.max(high_1d[i-19:i+1])
            donchian_low[i] = np.min(low_1d[i-19:i+1])
            donchian_mid[i] = (donchian_high[i] + donchian_low[i]) / 2
    
    # Align Donchian levels to 1d (already aligned since same timeframe)
    # But we still need to ensure proper handling for signal generation
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for Donchian and indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(trend_1w_up_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high, weekly trend up, volume spike
            if (close[i] > donchian_high[i] and trend_1w_up_aligned[i] and vol_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low, weekly trend down, volume spike
            elif (close[i] < donchian_low[i] and not trend_1w_up_aligned[i] and vol_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below Donchian mid or weekly trend turns down
            if (close[i] < donchian_mid[i] or not trend_1w_up_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above Donchian mid or weekly trend turns up
            if (close[i] > donchian_mid[i] or trend_1w_up_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals