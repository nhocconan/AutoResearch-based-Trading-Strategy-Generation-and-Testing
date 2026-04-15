#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian Breakout + 12h Volume Confirmation + 1d ADX Trend Filter
# Long when price breaks above Donchian(20) high, volume > 2x 20-bar median, and 1d ADX > 25 (trending).
# Short when price breaks below Donchian(20) low, volume > 2x 20-bar median, and 1d ADX > 25.
# Uses discrete position sizing (0.25) to limit trade frequency and avoid fee drag.
# Designed to capture strong trends in both bull and bear markets with volume confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = high_roll.values
    donchian_low = low_roll.values
    
    # Volume confirmation: > 2x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 2.0 * vol_median
    
    # 1-day ADX(14) for trend strength
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_ma = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    dm_plus_ma = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    dm_minus_ma = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # DI and DX
    di_plus = 100 * dm_plus_ma / (tr_ma + 1e-10)
    di_minus = 100 * dm_minus_ma / (tr_ma + 1e-10)
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    adx_1d = adx.values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_threshold[i]) or np.isnan(adx_1d_aligned[i])):
            continue
        
        # Long: breakout above Donchian high, volume spike, ADX > 25
        if (close[i] > donchian_high[i] and
            volume[i] > vol_threshold[i] and
            adx_1d_aligned[i] > 25):
            signals[i] = 0.25
        
        # Short: breakout below Donchian low, volume spike, ADX > 25
        elif (close[i] < donchian_low[i] and
              volume[i] > vol_threshold[i] and
              adx_1d_aligned[i] > 25):
            signals[i] = -0.25
        
        # Exit: price returns inside Donchian channel or ADX drops below 20
        elif (i > 0 and
              ((signals[i-1] == 0.25 and (close[i] < donchian_high[i] or adx_1d_aligned[i] < 20)) or
               (signals[i-1] == -0.25 and (close[i] > donchian_low[i] or adx_1d_aligned[i] < 20)))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_Donchian_12hVol_1dADX"
timeframe = "4h"
leverage = 1.0