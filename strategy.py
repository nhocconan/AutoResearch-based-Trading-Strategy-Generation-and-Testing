# 4h Donchian breakout + volume confirmation + volatility filter
# This strategy captures breakouts from price channels with volume confirmation
# and avoids whipsaws by requiring low volatility regime. Designed to work
# in both bull and bear markets by focusing on momentum bursts.
# Target: 20-50 trades per year to minimize fee drag.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_DonchianBreakout_Volume_VolatilityFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    # Volatility filter: ATR ratio < 1.2 (low volatility regime)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ratio = atr / (pd.Series(atr).rolling(window=50, min_periods=50).mean().values + 1e-10)
    volatility_filter = atr_ratio < 1.2
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(20, 50)  # Donchian and volatility filter
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or
            np.isnan(volume_filter[i]) or
            np.isnan(volatility_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        upper = donchian_high[i]
        lower = donchian_low[i]
        vol_ok = volume_filter[i]
        vol_filter = volatility_filter[i]
        
        if position == 0:
            # Enter long: break above upper Donchian + volume + low volatility
            if close[i] > upper and vol_ok and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: break below lower Donchian + volume + low volatility
            elif close[i] < lower and vol_ok and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below lower Donchian
            if close[i] < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above upper Donchian
            if close[i] > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals