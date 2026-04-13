#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout with 1d volume spike regime filter.
    # Long when price breaks above 20-period high with volume > 2.0x 20-period average.
    # Short when price breaks below 20-period low with volume > 2.0x 20-period average.
    # Exit when price crosses the 10-period EMA in opposite direction.
    # Uses volume spike to confirm institutional participation in breakouts.
    # Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume regime filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume spike regime: volume > 2.0x 20-period EMA
    vol_ema_20 = pd.Series(volume_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike_regime = volume_1d > (2.0 * vol_ema_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_regime)
    
    # Calculate 4h Donchian channels (20-period)
    high_ma_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h 10-period EMA for exit
    close_series = pd.Series(close)
    ema_10 = close_series.ewm(span=10, adjust=False, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(high_ma_20[i]) or np.isnan(low_ma_20[i]) or 
            np.isnan(ema_10[i]) or np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike regime filter from 1d
        volume_regime = volume_spike_aligned[i]
        
        # Breakout conditions
        long_breakout = high[i] > high_ma_20[i]
        short_breakout = low[i] < low_ma_20[i]
        
        # Exit conditions: price crosses 10 EMA in opposite direction
        long_exit = position == 1 and close[i] < ema_10[i]
        short_exit = position == -1 and close[i] > ema_10[i]
        
        # Fixed position size (discrete levels to minimize fee churn)
        position_size = 0.25
        
        # Entry conditions: only in volume spike regime
        if long_breakout and volume_regime and position != 1:
            position = 1
            signals[i] = position_size
        elif short_breakout and volume_regime and position != -1:
            position = -1
            signals[i] = -position_size
        # Exit conditions
        elif long_exit:
            position = 0
            signals[i] = 0.0
        elif short_exit:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_donchian_breakout_volume_spike_v1"
timeframe = "4h"
leverage = 1.0