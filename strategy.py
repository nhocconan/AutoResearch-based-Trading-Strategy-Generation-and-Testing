#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d volatility filter + volume confirmation
# - Long when price breaks above Donchian(20) high on 4h + 1d ATR(10) < 1.5 * ATR(30) (low vol regime) + volume > 1.5 * 20-period average
# - Short when price breaks below Donchian(20) low on 4h + 1d ATR(10) < 1.5 * ATR(30) + volume > 1.5 * 20-period average
# - Exit when price crosses back through Donchian(10) levels (faster exit) or volatility increases (ATR(10) > ATR(30))
# - Uses 4h for entry timing and 1d for volatility regime filter to avoid false breakouts
# - Target: 20-50 trades per year per symbol (80-200 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for volatility filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(10) and ATR(30) on 1d timeframe
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]  # First value
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    atr30 = pd.Series(tr).rolling(window=30, min_periods=30).mean().values
    
    # Align ATR to 4h timeframe
    atr10_4h = align_htf_to_ltf(prices, df_1d, atr10)
    atr30_4h = align_htf_to_ltf(prices, df_1d, atr30)
    
    # Volatility filter: low volatility regime (ATR10 < 1.5 * ATR30)
    vol_filter = atr10_4h < 1.5 * atr30_4h
    
    # Calculate Donchian channels on 4h timeframe
    high = prices['high'].values
    low = prices['low'].values
    
    # Donchian(20) for entry
    donch_high20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Donchian(10) for faster exit
    donch_high10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    donch_low10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # Volume confirmation
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup
        # Skip if NaN in indicators
        if np.isnan(donch_high20[i]) or np.isnan(donch_low20[i]) or \
           np.isnan(donch_high10[i]) or np.isnan(donch_low10[i]) or \
           np.isnan(vol_ma[i]) or np.isnan(vol_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        
        if position == 0:
            # Long entry: price breaks above Donchian(20) high + low vol + volume surge
            if price > donch_high20[i] and vol_filter[i] and vol > 1.5 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian(20) low + low vol + volume surge
            elif price < donch_low20[i] and vol_filter[i] and vol > 1.5 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below Donchian(10) high OR volatility increases
            if price < donch_high10[i] or not vol_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above Donchian(10) low OR volatility increases
            if price > donch_low10[i] or not vol_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_VolFilter_VolumeSurge"
timeframe = "4h"
leverage = 1.0