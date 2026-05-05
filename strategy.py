#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR-based volatility filter and volume confirmation
# Long when: price breaks above 20-period Donchian high, volume > 1.5x 20-period average, and 1d ATR(14) < 1d ATR(50) (low volatility regime)
# Short when: price breaks below 20-period Donchian low, volume > 1.5x 20-period average, and 1d ATR(14) < 1d ATR(50) (low volatility regime)
# Exit when price returns to the opposite Donchian level (mean reversion in low volatility) or breaks the mid-point
# Uses Donchian channels for structure, effective in both bull (breakout continuation) and bear (mean reversion via exits) markets.
# Timeframe: 4h, HTF: 1d. Target: 75-200 total trades over 4 years (19-50/year).

name = "4h_Donchian20_Breakout_1dATR_VolumeFilter"
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
    
    # Calculate volume confirmation on 4h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 1d data ONCE before loop for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(14) and ATR(50) for volatility regime filter
    if len(high_1d) >= 14:
        # True Range components
        tr1 = high_1d[1:] - low_1d[1:]
        tr2 = np.abs(high_1d[1:] - close_1d[:-1])
        tr3 = np.abs(low_1d[1:] - close_1d[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        # Prepend first TR as high-low for index 0
        tr_full = np.concatenate([[high_1d[0] - low_1d[0]], tr])
        
        atr_14 = pd.Series(tr_full).ewm(span=14, adjust=False, min_periods=14).mean().values
        atr_50 = pd.Series(tr_full).ewm(span=50, adjust=False, min_periods=50).mean().values
        
        # Low volatility regime: short-term ATR < long-term ATR
        vol_regime_filter = atr_14 < atr_50
    else:
        atr_14 = np.full(len(close_1d), np.nan)
        atr_50 = np.full(len(close_1d), np.nan)
        vol_regime_filter = np.zeros(len(close_1d), dtype=bool)
    
    # Align 1d volatility filter to 4h timeframe
    vol_regime_aligned = align_htf_to_ltf(prices, df_1d, vol_regime_filter)
    
    # Calculate 4h Donchian channels (20-period)
    if len(high) >= 20:
        donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_mid = (donchian_high + donchian_low) / 2
    else:
        donchian_high = np.full(n, np.nan)
        donchian_low = np.full(n, np.nan)
        donchian_mid = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(vol_regime_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high, volume filter, low volatility regime
            if (close[i] > donchian_high[i] and 
                open_price[i] <= donchian_high[i] and  # Ensure breakout happens on this bar
                volume_filter[i] and 
                vol_regime_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low, volume filter, low volatility regime
            elif (close[i] < donchian_low[i] and 
                  open_price[i] >= donchian_low[i] and  # Ensure breakdown happens on this bar
                  volume_filter[i] and 
                  vol_regime_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below Donchian mid (mean reversion) or breaks below Donchian low (reversal)
            if close[i] < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above Donchian mid (mean reversion) or breaks above Donchian high (reversal)
            if close[i] > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals