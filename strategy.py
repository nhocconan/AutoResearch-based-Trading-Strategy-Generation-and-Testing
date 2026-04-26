#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeTrend_Regime_v1
Hypothesis: 4h Donchian(20) breakout with volume confirmation and chop regime filter.
- Long when price breaks above Donchian(20) high with volume spike and CHOP > 61.8 (range regime)
- Short when price breaks below Donchian(20) low with volume spike and CHOP > 61.8 (range regime)
- Uses 4h timeframe targeting 75-200 total trades over 4 years (19-50/year)
- Volume confirmation: 2x 20-period average volume
- Chop regime filter: avoids trending markets where breakouts fail
- Designed for low trade frequency with proven edge on BTC/ETH from historical data
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for ATR calculation (for Chop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ATR(14) on 1d for Chop indicator
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate Chop indicator: Chop = 100 * log10(sum(atr1,14) / (max(high,14) - min(low,14))) / log10(14)
    max_high_14 = df_1d['high'].rolling(window=14, min_periods=14).max().values
    min_low_14 = df_1d['low'].rolling(window=14, min_periods=14).min().values
    sum_atr_14 = atr_14 * 14  # Approximation for sum of ATR over 14 periods
    range_14 = max_high_14 - min_low_14
    chop = 100 * np.log10(sum_atr_14 / range_14) / np.log10(14)
    chop = np.nan_to_num(chop, nan=50.0)  # Replace NaN with neutral value
    
    # Align Chop to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate Donchian(20) on 4h
    max_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    min_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume spike (20-period volume average on 4h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 2.0)  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for Donchian, 20 for volume MA)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(chop_aligned[i]) or 
            np.isnan(max_high_20[i]) or np.isnan(min_low_20[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Donchian breakout conditions with volume confirmation and chop regime filter
        price_above_donchian_high = close[i] > max_high_20[i]
        price_below_donchian_low = close[i] < min_low_20[i]
        chop_high = chop_aligned[i] > 61.8  # Range regime (mean reversion favorable)
        
        if position == 0:
            # Long: price breaks above Donchian high AND volume spike AND chop > 61.8
            if price_above_donchian_high and volume_spike[i] and chop_high:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND volume spike AND chop > 61.8
            elif price_below_donchian_low and volume_spike[i] and chop_high:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price falls below Donchian low OR chop < 38.2 (trending regime)
            if price_below_donchian_low or chop_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price rises above Donchian high OR chop < 38.2 (trending regime)
            if price_above_donchian_high or chop_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_VolumeTrend_Regime_v1"
timeframe = "4h"
leverage = 1.0