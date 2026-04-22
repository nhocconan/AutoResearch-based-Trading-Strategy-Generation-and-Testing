# Solution
#!/usr/bin/env python3
"""
Hypothesis: 12-hour Donchian channel breakout with 1-day volume spike and trend filter.
Long when price breaks above 20-period Donchian high with volume spike and 1-day EMA50 rising.
Short when price breaks below 20-period Donchian low with volume spike and 1-day EMA50 falling.
Exit when price returns to the Donchian midpoint or opposite breakout occurs.
Designed for low trade frequency by requiring volume confirmation and trend alignment.
Works in both bull and bear markets by following daily trend while using 12h Donchian for entries.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1-day data for EMA50 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Load 1-day data for volume spike detection - ONCE before loop
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Donchian channel (20-period) on 12h data
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_high = high_20
    donchian_low = low_20
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after enough data for Donchian
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Volume spike: current volume > 1.5 * 20-period MA
            volume_spike = volume[i] > 1.5 * vol_ma_20_aligned[i]
            
            # Long: Price breaks above Donchian high, volume spike, 1-day EMA50 rising
            if (close[i] > donchian_high[i] and 
                volume_spike and 
                ema50_1d_aligned[i] > ema50_1d_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low, volume spike, 1-day EMA50 falling
            elif (close[i] < donchian_low[i] and 
                  volume_spike and 
                  ema50_1d_aligned[i] < ema50_1d_aligned[i-1]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price returns to midpoint OR breaks below Donchian low
                if (close[i] <= donchian_mid[i] or 
                    close[i] < donchian_low[i]):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price returns to midpoint OR breaks above Donchian high
                if (close[i] >= donchian_mid[i] or 
                    close[i] > donchian_high[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian20_VolumeSpike_1dEMA50_Trend"
timeframe = "12h"
leverage = 1.0