#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) Breakout with Volume Spike and 1d Trend Filter (EMA50)
# Combines price channel breakout (Donchian) with volume confirmation and higher timeframe trend filter.
# Donchian breakouts capture momentum, volume confirms validity, daily EMA50 ensures trend alignment.
# Works in both bull/bear by only taking breakouts in direction of daily trend.
# Target: 20-30 trades/year (80-120 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Donchian channels (20-period) using prior period's values to avoid look-ahead
    # Upper = max(high[-21:-1]), Lower = min(low[-21:-1])
    high_shift = np.roll(high, 1)
    low_shift = np.roll(low, 1)
    high_shift[0] = np.nan
    low_shift[0] = np.nan
    
    donchian_upper = pd.Series(high_shift).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_shift).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume spike (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50  # for EMA50 and Donchian
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_50_1d[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Align daily indicators to 4h
        ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
        
        price = close[i]
        
        if position == 0:
            # Long breakout: price breaks above Donchian upper with volume spike and above daily EMA50
            if (price > donchian_upper[i] and vol_spike[i] and 
                price > ema_50_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short breakout: price breaks below Donchian lower with volume spike and below daily EMA50
            elif (price < donchian_lower[i] and vol_spike[i] and 
                  price < ema_50_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to Donchian lower or below daily EMA50
            if price < donchian_lower[i] or price < ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to Donchian upper or above daily EMA50
            if price > donchian_upper[i] or price > ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Donchian_Breakout_Volume_EMA50"
timeframe = "4h"
leverage = 1.0