#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA50 trend filter + volume confirmation (1.5x 20-bar avg)
# Long when price breaks above Donchian upper AND price > 1d EMA50 AND volume > 1.5x 20-bar average
# Short when price breaks below Donchian lower AND price < 1d EMA50 AND volume > 1.5x 20-bar average
# Exit when price crosses Donchian midpoint (mean of upper/lower) OR EMA50 trend flip
# Donchian provides clear structure, EMA50 filters counter-trend trades, volume confirms conviction
# Target: 20-50 trades/year per symbol (80-200 total over 4 years) for 4h timeframe
# Discrete sizing (0.25) to limit fee drag

name = "4h_Donchian20_1dEMA50_Trend_VolumeConfirm"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA50 on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian(20) channels
    if len(high) >= 20:
        donch_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donch_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donch_mid = (donch_upper + donch_lower) / 2.0
    else:
        donch_upper = np.full(n, np.nan)
        donch_lower = np.full(n, np.nan)
        donch_mid = np.full(n, np.nan)
    
    # Volume confirmation: volume > 1.5x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(donch_upper[i]) or 
            np.isnan(donch_lower[i]) or 
            np.isnan(donch_mid[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above Donchian upper AND price > 1d EMA50 AND volume spike
            if (close[i] > donch_upper[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: break below Donchian lower AND price < 1d EMA50 AND volume spike
            elif (close[i] < donch_lower[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price < Donchian midpoint OR price < 1d EMA50 (trend flip)
            if (close[i] < donch_mid[i] or 
                close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price > Donchian midpoint OR price > 1d EMA50 (trend flip)
            if (close[i] > donch_mid[i] or 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals