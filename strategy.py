#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR volatility filter and volume confirmation
# Long when: price breaks above 20-period Donchian high, volume > 1.5x 20-period average, and 1d ATR(14) < 0.03*close (low volatility environment)
# Short when: price breaks below 20-period Donchian low, volume > 1.5x 20-period average, and 1d ATR(14) < 0.03*close
# Exit when price returns to the midpoint of the Donchian channel (mean reversion)
# Uses Donchian structure for breakouts in trending markets and mean reversion in ranging markets, with volatility filter to avoid false breakouts during high volatility.
# Timeframe: 4h, HTF: 1d. Target: 75-200 total trades over 4 years (19-50/year).

name = "4h_Donchian20_Breakout_1dATRFilter_VolumeConfirm"
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
    open_price = prices['open'].values
    
    # Calculate Donchian channel on 4h (20-period)
    if len(high) >= 20:
        donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_mid = (donchian_high + donchian_low) / 2
    else:
        donchian_high = np.full(n, np.nan)
        donchian_low = np.full(n, np.nan)
        donchian_mid = np.full(n, np.nan)
    
    # Calculate volume confirmation on 4h (20-period average)
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
        tr1 = np.abs(high_1d[1:] - low_1d[:-1])
        tr2 = np.abs(high_1d[1:] - close_1d[:-1])
        tr3 = np.abs(low_1d[1:] - close_1d[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
        # Normalize ATR by price to get volatility percentage
        volatility_filter = atr_14 < (0.03 * close_1d)
    else:
        atr_14 = np.full(len(close_1d), np.nan)
        volatility_filter = np.zeros(len(close_1d), dtype=bool)
    
    # Align 1d indicators to 4h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    volatility_filter_aligned = align_htf_to_ltf(prices, df_1d, volatility_filter.astype(float)) > 0.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or 
            np.isnan(atr_14_aligned[i]) or 
            np.isnan(volatility_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high, volume filter, low volatility
            if (close[i] > donchian_high[i] and 
                open_price[i] <= donchian_high[i] and  # Ensure breakout happens on this bar
                volume_filter[i] and 
                volatility_filter_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low, volume filter, low volatility
            elif (close[i] < donchian_low[i] and 
                  open_price[i] >= donchian_low[i] and  # Ensure breakdown happens on this bar
                  volume_filter[i] and 
                  volatility_filter_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian midpoint (mean reversion)
            if close[i] < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian midpoint (mean reversion)
            if close[i] > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals