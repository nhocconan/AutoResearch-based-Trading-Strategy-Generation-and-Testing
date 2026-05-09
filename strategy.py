#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 1d ATR filter and volume confirmation
# Long when price breaks above 20-period high with 1d ATR < 30-period average (low volatility breakout)
# Short when price breaks below 20-period low with 1d ATR < 30-period average and volume spike
# Exit when price returns to 20-period midpoint or reverses to opposite band
# Designed to capture low-volatility breakouts in both trending and ranging markets
# Target: 100-180 total trades over 4 years (25-45/year) with size 0.25

name = "4h_Donchian20_ATRFilter_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d ATR for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First period has no previous close
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(30) - 30-period ATR
    atr_30 = pd.Series(tr).rolling(window=30, min_periods=30).mean().values
    atr_7 = pd.Series(tr).rolling(window=7, min_periods=7).mean().values
    atr_ratio = atr_7 / atr_30  # Short-term ATR relative to long-term ATR
    
    # ATR ratio < 1.0 indicates low volatility (7-day ATR < 30-day ATR)
    low_vol_filter = atr_ratio < 1.0
    low_vol_filter_aligned = align_htf_to_ltf(prices, df_1d, low_vol_filter)
    
    # Donchian channels (20-period) on 4h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Need enough data for ATR calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(low_vol_filter_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Donchian high, low volatility, volume spike
            if (close[i] > donchian_high[i] and 
                low_vol_filter_aligned[i] and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low, low volatility, volume spike
            elif (close[i] < donchian_low[i] and 
                  low_vol_filter_aligned[i] and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to midpoint or breaks below low
            if (close[i] <= donchian_mid[i]) or (close[i] < donchian_low[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to midpoint or breaks above high
            if (close[i] >= donchian_mid[i]) or (close[i] > donchian_high[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals