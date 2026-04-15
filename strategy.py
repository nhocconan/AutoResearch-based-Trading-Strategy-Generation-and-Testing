#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian Breakout + 1d ADX Trend + Volume Spike
# Long when price breaks above 4h Donchian(20) high and 1d ADX > 25 (trending).
# Short when price breaks below 4h Donchian(20) low and 1d ADX > 25.
# Volume confirmation requires > 2x 20-bar median volume.
# Exit on opposite Donchian breakout or ADX < 20 (range).
# Conservative sizing (0.25) to limit trade frequency and avoid fee drag.
# Works in bull (breakouts) and bear (breakdowns) markets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day ADX(14) for trend strength
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr14 = pd.Series(tr).ewm(alpha=1/14, adjust=False).values
    dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False).values
    dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False).values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / (tr14 + 1e-10)
    di_minus = 100 * dm_minus_14 / (tr14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 4h Donchian channels (20-period)
    def donchian_channel(high, low, window):
        upper = pd.Series(high).rolling(window=window, min_periods=1).max()
        lower = pd.Series(low).rolling(window=window, min_periods=1).min()
        return upper.values, lower.values
    
    donchian_high, donchian_low = donchian_channel(high, low, 20)
    
    # Volume confirmation: current > 2x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 2.0 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Long: price breaks above Donchian high, ADX > 25 (trending), volume spike
        if (close[i] > donchian_high[i-1] and 
            adx_1d_aligned[i] > 25 and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short: price breaks below Donchian low, ADX > 25 (trending), volume spike
        elif (close[i] < donchian_low[i-1] and 
              adx_1d_aligned[i] > 25 and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit: opposite breakout or ADX < 20 (range market)
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and (close[i] < donchian_low[i-1] or adx_1d_aligned[i] < 20)) or
               (signals[i-1] == -0.25 and (close[i] > donchian_high[i-1] or adx_1d_aligned[i] < 20)))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_Donchian_ADX1d_Volume"
timeframe = "4h"
leverage = 1.0