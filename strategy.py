#!/usr/bin/env python3
"""
Hypothesis: 6-hour Donchian breakout with 12-hour ADX trend filter and volume confirmation.
Long when price breaks above Donchian(20) high with ADX > 20 and volume > 1.5x average.
Short when price breaks below Donchian(20) low with ADX > 20 and volume > 1.5x average.
Exit when price returns to Donchian midpoint or ADX falls below 15.
Designed for 6h timeframe to capture medium-term trends with reduced whipsaws.
Works in bull markets via breakouts and in bear markets via short breakdowns.
"""

import numpy as np
import pandas as pd
from mta_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 6h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = highest_high.values
    donchian_low = lowest_low.values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate volume average (20-period)
    volume_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 12h data for ADX - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12-hour ADX (14-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_12h - np.roll(high_12h, 1)) > (np.roll(low_12h, 1) - low_12h),
                       np.maximum(high_12h - np.roll(high_12h, 1), 0), 0)
    dm_minus = np.where((np.roll(low_12h, 1) - low_12h) > (high_12h - np.roll(high_12h, 1)),
                        np.maximum(np.roll(low_12h, 1) - low_12h, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    dm_plus14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum()
    dm_minus14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum()
    
    # Directional Indicators
    plus_di = 100 * dm_plus14 / tr14
    minus_di = 100 * dm_minus14 / tr14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean()
    adx_values = adx.values
    
    # Align HTF ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx_values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(volume_avg[i]) or np.isnan(adx_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price above Donchian high with trend and volume confirmation
            if (close[i] > donchian_high[i] and 
                adx_aligned[i] > 20 and 
                volume[i] > 1.5 * volume_avg[i]):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price below Donchian low with trend and volume confirmation
            elif (close[i] < donchian_low[i] and 
                  adx_aligned[i] > 20 and 
                  volume[i] > 1.5 * volume_avg[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price returns to midpoint or trend weakens
                if close[i] <= donchian_mid[i] or adx_aligned[i] < 15:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price returns to midpoint or trend weakens
                if close[i] >= donchian_mid[i] or adx_aligned[i] < 15:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Donchian_Breakout_ADX_Volume"
timeframe = "6h"
leverage = 1.0