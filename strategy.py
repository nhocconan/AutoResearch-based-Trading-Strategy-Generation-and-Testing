#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian channel breakout with 1-day ATR filter and volume confirmation
# Long when price breaks above 20-period Donchian high AND 1-day ATR > 20-period ATR average AND volume > 1.5x volume average
# Short when price breaks below 20-period Donchian low AND 1-day ATR > 20-period ATR average AND volume > 1.5x volume average
# Exit when price crosses back inside the Donchian channel (opposite band)
# Uses Donchian channels to capture breakouts, ATR to filter for volatile regimes, volume for confirmation
# Target: 75-200 total trades over 4 years (19-50/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for ATR filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Donchian Channels on 4h (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 1-day ATR (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # First value has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_avg = pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_14_avg_aligned = align_htf_to_ltf(prices, df_1d, atr_14_avg)
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (20 for Donchian + buffer)
    start = 30
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(atr_14_avg_aligned[i]) or 
            np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: price breaks above Donchian high AND ATR > ATR average AND volume confirmation
            if (price > donchian_high[i] and atr_14_aligned[i] > atr_14_avg_aligned[i] and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: price breaks below Donchian low AND ATR > ATR average AND volume confirmation
            elif (price < donchian_low[i] and atr_14_aligned[i] > atr_14_avg_aligned[i] and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses back inside Donchian channel (below Donchian low)
            if price < donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses back inside Donchian channel (above Donchian high)
            if price > donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Donchian_1dATR_Volume"
timeframe = "4h"
leverage = 1.0