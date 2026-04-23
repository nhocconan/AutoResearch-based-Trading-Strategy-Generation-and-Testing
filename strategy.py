#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and volume confirmation
- Donchian(20) on 6h captures medium-term price channels and breakouts
- Weekly Camarilla R4/S4 levels (from 1w) determine major trend direction: price above weekly pivot = bullish bias, below = bearish bias
- Volume > 1.8x 20-period average confirms breakout strength
- Designed for 6h timeframe targeting 12-37 trades/year (50-150 over 4 years) to minimize fee drag
- Works in bull markets via breakouts with weekly trend, in bear markets via mean reversion at extreme weekly levels
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
    
    # Get weekly data for Camarilla pivot levels (trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Previous week's OHLC for Camarilla calculation
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    prev_week_close = df_1w['close'].shift(1).values
    
    # Weekly Camarilla R4, S4 levels: R4 = close + (high-low)*1.1/2, S4 = close - (high-low)*1.1/2
    weekly_r4 = prev_week_close + (prev_week_high - prev_week_low) * 1.1 / 2
    weekly_s4 = prev_week_close - (prev_week_high - prev_week_low) * 1.1 / 2
    weekly_pivot = (prev_week_high + prev_week_low + prev_week_close) / 3
    
    # Align weekly levels to 6h timeframe (completed 1w bar only)
    weekly_r4_aligned = align_htf_to_ltf(prices, df_1w, weekly_r4)
    weekly_s4_aligned = align_htf_to_ltf(prices, df_1w, weekly_s4)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Donchian(20) on 6h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 1)  # Donchian20, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i]) or
            np.isnan(weekly_r4_aligned[i]) or 
            np.isnan(weekly_s4_aligned[i]) or 
            np.isnan(weekly_pivot_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend bias
        weekly_bullish = close[i] > weekly_pivot_aligned[i]
        weekly_bearish = close[i] < weekly_pivot_aligned[i]
        
        # Donchian breakout signals with weekly trend filter and volume confirmation
        # Long: price breaks above Donchian high + weekly bullish bias + volume spike
        # Short: price breaks below Donchian low + weekly bearish bias + volume spike
        long_signal = (close[i] > donchian_high[i] and 
                      weekly_bullish and
                      volume[i] > 1.8 * vol_ma[i])
        
        short_signal = (close[i] < donchian_low[i] and 
                       weekly_bearish and
                       volume[i] > 1.8 * vol_ma[i])
        
        if position == 0:
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
                # Exit long: price breaks below Donchian low or weekly turns bearish
                if (close[i] < donchian_low[i] or 
                    close[i] < weekly_pivot_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: price breaks above Donchian high or weekly turns bullish
                if (close[i] > donchian_high[i] or 
                    close[i] > weekly_pivot_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Donchian20_WeeklyCamarillaR4S4_PivotTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0