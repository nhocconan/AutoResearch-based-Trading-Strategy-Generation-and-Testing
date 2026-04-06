#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout + 12h ADX trend filter + volume confirmation
# Long when price breaks above Donchian(20) high AND ADX(12h) > 25 AND volume > 1.5x avg
# Short when price breaks below Donchian(20) low AND ADX(12h) > 25 AND volume > 1.5x avg
# Exit when price returns to Donchian midpoint OR ADX < 20 (trend weakening)
# Uses 4h timeframe with 12h trend filter to avoid whipsaws, targeting 100-200 total trades over 4 years
# Works in bull/bear by only trading strong trends (ADX filter)

name = "4h_donchian_adx_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = highest_high.values
    donchian_low = lowest_low.values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # ADX (12-period) from 12h timeframe for trend strength
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high_12h[0] - low_12h[0]  # first period
    
    # Directional Movement
    dm_plus = np.where((high_12h - np.roll(high_12h, 1)) > (np.roll(low_12h, 1) - low_12h), 
                       np.maximum(high_12h - np.roll(high_12h, 1), 0), 0)
    dm_minus = np.where((np.roll(low_12h, 1) - low_12h) > (high_12h - np.roll(high_12h, 1)), 
                        np.maximum(np.roll(low_12h, 1) - low_12h, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr14 = pd.Series(tr).rolling(window=12, min_periods=12).sum()
    dm_plus14 = pd.Series(dm_plus).rolling(window=12, min_periods=12).sum()
    dm_minus14 = pd.Series(dm_minus).rolling(window=12, min_periods=12).sum()
    
    # DI and DX
    di_plus = 100 * dm_plus14 / (tr14 + 1e-10)
    di_minus = 100 * dm_minus14 / (tr14 + 1e-10)
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    
    # ADX (smoothed DX)
    adx = pd.Series(dx).rolling(window=12, min_periods=12).mean()
    adx_values = adx.values
    
    # Align 12h ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx_values)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(25, n):
        # Skip if required data not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(adx_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: price returns to midpoint OR trend weakens (ADX < 20)
        if position == 1:  # long position
            if close[i] <= donchian_mid[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= donchian_mid[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + strong trend (ADX > 25) + volume confirmation
            # Long: price breaks above Donchian high
            if (close[i] > donchian_high[i] and adx_aligned[i] > 25 and volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low
            elif (close[i] < donchian_low[i] and adx_aligned[i] > 25 and volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals