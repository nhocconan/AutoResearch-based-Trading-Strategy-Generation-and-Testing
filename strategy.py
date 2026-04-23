#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation
- Donchian(20) provides clear breakout levels adapted to recent volatility
- Weekly Camarilla pivot levels (R4/S4) determine the major trend direction from higher timeframe
- Only trade breakouts in alignment with weekly trend to avoid counter-trend whipsaws
- Volume confirmation (> 2.0x 20-period average) ensures breakout has conviction
- Designed for 6h timeframe targeting 12-37 trades/year (50-150 over 4 years)
- Weekly pivot filter adds structural bias that works in both bull and bear markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for weekly Camarilla pivot levels (trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla pivot levels (R4, S4) for trend filter
    # Based on prior 1w bar's OHLC
    typical_price_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    
    camarilla_r4_1w = typical_price_1w + (range_1w * 1.1 / 2.0)  # R4 = C + (H-L)*1.1/2
    camarilla_s4_1w = typical_price_1w - (range_1w * 1.1 / 2.0)  # S4 = C - (H-L)*1.1/2
    
    # Align weekly Camarilla levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4_1w)
    s4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4_1w)
    
    # Calculate 6h Donchian(20) channels
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 20)  # Donchian, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(high_ma[i]) or np.isnan(low_ma[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine breakout conditions
        # Long: price breaks above upper Donchian with weekly bullish bias and volume
        # Short: price breaks below lower Donchian with weekly bearish bias and volume
        price_above_donchian = close[i] > high_ma[i]
        price_below_donchian = close[i] < low_ma[i]
        
        # Weekly trend filter: price > weekly R4 for bullish bias, price < weekly S4 for bearish bias
        weekly_bullish = close[i] > r4_aligned[i]
        weekly_bearish = close[i] < s4_aligned[i]
        
        if position == 0:
            # Long conditions: price breaks above upper Donchian, weekly bullish, volume spike
            long_signal = (price_above_donchian and 
                          weekly_bullish and
                          volume[i] > 2.0 * vol_ma[i])
            
            # Short conditions: price breaks below lower Donchian, weekly bearish, volume spike
            short_signal = (price_below_donchian and 
                           weekly_bearish and
                           volume[i] > 2.0 * vol_ma[i])
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite Donchian break or weekly trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: price falls below lower Donchian or weekly turns bearish
                if (price_below_donchian or 
                    not weekly_bullish):  # Weekly trend reversal
                    exit_signal = True
            elif position == -1:
                # Exit short: price rises above upper Donchian or weekly turns bullish
                if (price_above_donchian or 
                    not weekly_bearish):  # Weekly trend reversal
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Donchian20_WeeklyCamarillaR4S4_Trend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0