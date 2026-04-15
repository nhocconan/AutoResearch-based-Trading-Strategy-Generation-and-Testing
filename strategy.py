#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + ADX trend filter
# Trades breakouts above 20-period high or below 20-period low with volume > 1.5x median and ADX > 25
# Works in bull markets (up breakouts) and bear markets (down breakouts)
# Target: 100-200 total trades over 4 years (25-50/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ADX(14) for trend filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(close, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(close, 1)), 
                        np.maximum(np.roll(close, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(adx[i])):
            continue
        
        # Long entry: break above Donchian high + volume confirmation + ADX > 25
        if (close[i] > donchian_high[i] and
            volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
            adx[i] > 25 and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: break below Donchian low + volume confirmation + ADX > 25
        elif (close[i] < donchian_low[i] and
              volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
              adx[i] > 25 and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse breakout or ADX < 20 (ranging market)
        elif position == 1 and (close[i] < donchian_low[i] or adx[i] < 20):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > donchian_high[i] or adx[i] < 20):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_Volume_ADX"
timeframe = "4h"
leverage = 1.0