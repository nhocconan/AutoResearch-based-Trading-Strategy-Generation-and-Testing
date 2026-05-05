#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d EMA50 trend filter + volume confirmation
# Long when: price > upper Donchian(20) AND 1d EMA50 uptrend (close > EMA50) AND volume > 1.5x 20-period MA
# Short when: price < lower Donchian(20) AND 1d EMA50 downtrend (close < EMA50) AND volume > 1.5x 20-period MA
# Exit when: price crosses opposite Donchian band OR volume < 1.2x 20-period MA (weak conviction)
# Uses Donchian for structure, 1d EMA50 for HTF trend regime, volume for conviction
# Timeframe: 12h, HTF: 1d for EMA50 trend. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "12h_Donchian20_1dEMA50_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian(20) channels on 12h
    if len(high) >= 20:
        upper_donchian = pd.Series(high).rolling(window=20, min_periods=20).max().values
        lower_donchian = pd.Series(low).rolling(window=20, min_periods=20).min().values
    else:
        upper_donchian = np.full(n, np.nan)
        lower_donchian = np.full(n, np.nan)
    
    # Volume confirmation on 12h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
        volume_weak = volume < (1.2 * vol_ma_20)  # for exit condition
    else:
        volume_filter = np.zeros(n, dtype=bool)
        volume_weak = np.zeros(n, dtype=bool)
    
    # Get 1d data ONCE before loop for EMA50 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # need sufficient data for EMA50
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d
    close_1d = df_1d['close'].values
    if len(close_1d) >= 50:
        ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    else:
        ema_50_1d = np.full(len(df_1d), np.nan)
    
    # 1d EMA50 trend: close > EMA50 = uptrend, close < EMA50 = downtrend
    ema_uptrend = close_1d > ema_50_1d
    ema_downtrend = close_1d < ema_50_1d
    
    # Align 1d EMA50 trend to 12h timeframe
    ema_uptrend_aligned = align_htf_to_ltf(prices, df_1d, ema_uptrend.astype(float))
    ema_downtrend_aligned = align_htf_to_ltf(prices, df_1d, ema_downtrend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(upper_donchian[i]) or np.isnan(lower_donchian[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(volume_weak[i]) or 
            np.isnan(ema_uptrend_aligned[i]) or np.isnan(ema_downtrend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price > upper Donchian + 1d EMA50 uptrend + volume filter
            if (close[i] > upper_donchian[i] and 
                ema_uptrend_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price < lower Donchian + 1d EMA50 downtrend + volume filter
            elif (close[i] < lower_donchian[i] and 
                  ema_downtrend_aligned[i] == 1.0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price < lower Donchian OR volume weak
            if (close[i] < lower_donchian[i] or volume_weak[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price > upper Donchian OR volume weak
            if (close[i] > upper_donchian[i] or volume_weak[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals