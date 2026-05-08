#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h EMA trend filter, 4h Donchian(20) breakout, and volume confirmation.
# Long when 12h EMA > 0 (uptrend), price breaks above 4h Donchian upper band, volume > 1.3x average.
# Short when 12h EMA < 0 (downtrend), price breaks below 4h Donchian lower band, volume > 1.3x average.
# Position sizing fixed at 0.25. Max 400 trades over 4 years (100/year) to avoid fee drag.
# Works in bull (trend follow) and bear (trend still exists in downtrends).

name = "4h_12hEMA_4hDonchian_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Get 4h data for Donchian bands
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 12h EMA(50) - trend filter
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_slope = ema_12h - np.roll(ema_12h, 1)
    ema_12h_slope[0] = 0
    ema_trend_up = ema_12h_slope > 0
    ema_trend_down = ema_12h_slope < 0
    
    # 4h Donchian(20) bands
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align 12h EMA trend to 4h
    ema_trend_up_aligned = align_htf_to_ltf(prices, df_12h, ema_trend_up.astype(float))
    ema_trend_down_aligned = align_htf_to_ltf(prices, df_12h, ema_trend_down.astype(float))
    # Align 4h Donchian bands to 4h
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_bar = 0
    
    start_idx = 70  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_trend_up_aligned[i]) or np.isnan(ema_trend_down_aligned[i]) or
            np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 12h EMA up, price breaks above 4h Donchian upper band, volume spike
            if (ema_trend_up_aligned[i] and
                close[i] > donchian_high_aligned[i] and
                vol_ratio[i] > 1.3):
                signals[i] = 0.25
                position = 1
                entry_bar = i
            # Short: 12h EMA down, price breaks below 4h Donchian lower band, volume spike
            elif (ema_trend_down_aligned[i] and
                  close[i] < donchian_low_aligned[i] and
                  vol_ratio[i] > 1.3):
                signals[i] = -0.25
                position = -1
                entry_bar = i
        elif position == 1:
            # Long exit: EMA trend down, price breaks below Donchian lower band, or max 20 bars held
            if (ema_trend_down_aligned[i] or 
                close[i] < donchian_low_aligned[i] or
                i - entry_bar >= 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: EMA trend up, price breaks above Donchian upper band, or max 20 bars held
            if (ema_trend_up_aligned[i] or 
                close[i] > donchian_high_aligned[i] or
                i - entry_bar >= 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals