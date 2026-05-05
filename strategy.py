#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h EMA(50) trend filter + volume confirmation
# Long when: price breaks above Donchian(20) upper band AND 12h EMA(50) uptrend (price > EMA) AND volume > 1.5x 20-period MA
# Short when: price breaks below Donchian(20) lower band AND 12h EMA(50) downtrend (price < EMA) AND volume > 1.5x 20-period MA
# Exit when: price returns to Donchian(20) midpoint OR volume drops below average
# Uses Donchian for structure, 12h EMA for higher-timeframe trend, volume for conviction
# Timeframe: 4h, HTF: 12h for EMA trend. Target: 75-200 total trades over 4 years (19-50/year) to avoid fee drag.

name = "4h_Donchian20_12hEMA50_VolumeConfirm"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian(20) channels
    if len(high) >= 20:
        donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_mid = (donchian_upper + donchian_lower) / 2
    else:
        donchian_upper = np.full(n, np.nan)
        donchian_lower = np.full(n, np.nan)
        donchian_mid = np.full(n, np.nan)
    
    # Volume confirmation on 4h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 12h data ONCE before loop for EMA calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # need sufficient data for EMA
        return np.zeros(n)
    
    # Calculate EMA(50) on 12h
    close_12h = df_12h['close'].values
    if len(close_12h) >= 50:
        ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    else:
        ema_50_12h = np.full(len(df_12h), np.nan)
    
    # Determine 12h trend: price > EMA = uptrend, price < EMA = downtrend
    ema_trend_up = close_12h > ema_50_12h  # uptrend
    ema_trend_down = close_12h < ema_50_12h  # downtrend
    
    # Align 12h EMA trend to 4h timeframe
    ema_trend_up_aligned = align_htf_to_ltf(prices, df_12h, ema_trend_up.astype(float))
    ema_trend_down_aligned = align_htf_to_ltf(prices, df_12h, ema_trend_down.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(volume_filter[i]) or 
            np.isnan(ema_trend_up_aligned[i]) or np.isnan(ema_trend_down_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above upper band + 12h uptrend + volume filter
            if (close[i] > donchian_upper[i] and 
                ema_trend_up_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: break below lower band + 12h downtrend + volume filter
            elif (close[i] < donchian_lower[i] and 
                  ema_trend_down_aligned[i] == 1.0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to midpoint OR volume drops below average
            if (close[i] <= donchian_mid[i] or not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to midpoint OR volume drops below average
            if (close[i] >= donchian_mid[i] or not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals