#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA50 trend filter + volume confirmation
# Long when: price breaks above Donchian(20) high AND price > 1w EMA50 AND volume > 1.5x 20-period MA
# Short when: price breaks below Donchian(20) low AND price < 1w EMA50 AND volume > 1.5x 20-period MA
# Exit when: price crosses Donchian(20) midpoint OR EMA50 trend reverses
# Uses Donchian for structure, 1w EMA for major trend filter, volume for conviction
# Timeframe: 1d, HTF: 1w. Target: 30-100 total trades over 4 years (7-25/year) to avoid fee drag.

name = "1d_Donchian20_1wEMA50_VolumeConfirm"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian(20) on 1d
    if len(high) >= 20:
        donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donch_mid = (donch_high + donch_low) / 2
    else:
        donch_high = np.full(n, np.nan)
        donch_low = np.full(n, np.nan)
        donch_mid = np.full(n, np.nan)
    
    # Get 1w data ONCE before loop for EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1w
    close_1w = df_1w['close'].values
    if len(close_1w) >= 50:
        ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    else:
        ema_50 = np.full(len(close_1w), np.nan)
    
    # Align 1w EMA50 to 1d timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Volume confirmation on 1d
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(donch_mid[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above Donchian high + price > 1w EMA50 + volume filter
            if (close[i] > donch_high[i] and 
                close[i] > ema_50_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: break below Donchian low + price < 1w EMA50 + volume filter
            elif (close[i] < donch_low[i] and 
                  close[i] < ema_50_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses Donchian midpoint OR price < 1w EMA50
            if (close[i] < donch_mid[i] or close[i] < ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses Donchian midpoint OR price > 1w EMA50
            if (close[i] > donch_mid[i] or close[i] > ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals