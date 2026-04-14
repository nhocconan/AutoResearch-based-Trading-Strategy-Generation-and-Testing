#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian channel breakout with 1-day ATR filter and volume confirmation
# Long when price breaks above 20-period Donchian high AND ATR(14) > 1.5x ATR(50) AND volume > 1.5x 20-period average
# Short when price breaks below 20-period Donchian low AND ATR(14) > 1.5x ATR(50) AND volume > 1.5x 20-period average
# Exit when price crosses back inside the Donchian channel (opposite side)
# Uses Donchian channels for breakout signals, ATR ratio to filter for volatile breakouts, volume for confirmation
# Target: 75-200 total trades over 4 years (19-50/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1-day data ONCE before loop for ATR filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Donchian channels on 4h (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR on 1d for volatility filter (ATR(14) and ATR(50))
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) and ATR(50)
    atr14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr50_1d = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # ATR ratio filter: ATR(14) > 1.5 * ATR(50) indicates elevated volatility
    atr_ratio = atr14_1d / atr50_1d
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (max of 20, 50)
    start = 50
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr_ratio_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: break above Donchian high + high volatility + volume confirmation
            if (price > donchian_high[i] and atr_ratio_aligned[i] > 1.5 and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: break below Donchian low + high volatility + volume confirmation
            elif (price < donchian_low[i] and atr_ratio_aligned[i] > 1.5 and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses back below Donchian low
            if price < donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses back above Donchian high
            if price > donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Donchian_ATR_Volume_Filter"
timeframe = "4h"
leverage = 1.0