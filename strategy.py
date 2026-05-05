#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR filter and volume confirmation
# Long when: price breaks above Donchian(20) high, volume > 1.5x 20-period average, and 1d ATR(14) < 0.03*close (low volatility breakout)
# Short when: price breaks below Donchian(20) low, volume > 1.5x 20-period average, and 1d ATR(14) < 0.03*close
# Exit when price returns to Donchian(20) midpoint (mean reversion)
# Uses Donchian channels for structure, effective in both bull (breakout continuation) and bear (mean reversion via exits) markets.
# Timeframe: 4h, HTF: 1d. Target: 75-200 total trades over 4 years (19-50/year).

name = "4h_Donchian20_Breakout_1dATRFilter_VolumeConfirm"
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
    open_price = prices['open'].values
    
    # Calculate Donchian(20) channels on 4h
    if len(high) >= 20:
        donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_mid = (donchian_high + donchian_low) / 2
    else:
        donchian_high = np.full(n, np.nan)
        donchian_low = np.full(n, np.nan)
        donchian_mid = np.full(n, np.nan)
    
    # Calculate volume confirmation on 4h using 20-period MA
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 1d data ONCE before loop for ATR filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(14)
    if len(high_1d) >= 14:
        tr1 = pd.Series(high_1d).diff().abs()
        tr2 = (pd.Series(high_1d) - pd.Series(close_1d).shift()).abs()
        tr3 = (pd.Series(low_1d) - pd.Series(close_1d).shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_14 = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
        # Low volatility filter: ATR < 3% of price
        atr_filter = atr_14 < (0.03 * close_1d)
    else:
        atr_14 = np.full(len(close_1d), np.nan)
        atr_filter = np.zeros(len(close_1d), dtype=bool)
    
    # Align 1d ATR filter to 4h timeframe
    atr_filter_aligned = align_htf_to_ltf(prices, df_1d, atr_filter)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or 
            np.isnan(volume_filter[i]) or 
            np.isnan(atr_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high, volume filter, low volatility
            if (close[i] > donchian_high[i] and 
                open_price[i] <= donchian_high[i] and  # Ensure breakout happens on this bar
                volume_filter[i] and 
                atr_filter_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low, volume filter, low volatility
            elif (close[i] < donchian_low[i] and 
                  open_price[i] >= donchian_low[i] and  # Ensure breakdown happens on this bar
                  volume_filter[i] and 
                  atr_filter_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below Donchian midpoint (mean reversion)
            if close[i] < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above Donchian midpoint (mean reversion)
            if close[i] > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals