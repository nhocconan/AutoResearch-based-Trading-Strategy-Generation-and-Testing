#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d EMA(34) trend filter, 4h Donchian(20) breakout, and volume confirmation.
# Long when 1d EMA(34) up, price breaks above 4h Donchian upper band, volume > 1.5x average.
# Short when 1d EMA(34) down, price breaks below 4h Donchian lower band, volume > 1.5x average.
# Fixed position size of 0.25 to limit risk and trade frequency.
# Target: 20-50 total trades over 4 years (5-12/year) to avoid fee drag.
# Works in bull (trend follow) and bear (trend still exists in downtrends).

name = "4h_1dEMA34_4hDonchian_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA(34) trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Get 4h data for Donchian bands
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 1d EMA(34) - upward if current > previous
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_up = ema_34_1d > np.roll(ema_34_1d, 1)
    ema_up[0] = False  # First value has no previous
    
    # 4h Donchian(20) bands
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align 1d EMA trend to 4h
    ema_up_aligned = align_htf_to_ltf(prices, df_1d, ema_up.astype(float))
    # Align 4h Donchian bands to 4h
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_bar = 0
    
    start_idx = 40  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_up_aligned[i]) or np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 1d EMA up, price breaks above 4h Donchian upper band, volume spike
            if (ema_up_aligned[i] and
                close[i] > donchian_high_aligned[i] and
                vol_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
                entry_bar = i
            # Short: 1d EMA down, price breaks below 4h Donchian lower band, volume spike
            elif (not ema_up_aligned[i] and
                  close[i] < donchian_low_aligned[i] and
                  vol_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
                entry_bar = i
        elif position == 1:
            # Long exit: EMA down, price breaks below Donchian lower band, or max 10 bars held
            if (not ema_up_aligned[i] or 
                close[i] < donchian_low_aligned[i] or
                i - entry_bar >= 10):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: EMA up, price breaks above Donchian upper band, or max 10 bars held
            if (ema_up_aligned[i] or 
                close[i] > donchian_high_aligned[i] or
                i - entry_bar >= 10):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals