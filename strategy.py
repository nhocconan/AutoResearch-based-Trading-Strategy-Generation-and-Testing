#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Donchian(20) breakout + 1d Williams %R extreme + volume confirmation
    # Long when: price breaks above Donchian(20) high AND Williams %R(1d) < -80 (oversold) AND volume > 1.5x avg volume
    # Short when: price breaks below Donchian(20) low AND Williams %R(1d) > -20 (overbought) AND volume > 1.5x avg volume
    # Exit when: price crosses Donchian midpoint OR Williams %R returns to neutral range (-80 to -20)
    # Uses discrete sizing (0.25) targeting 50-150 trades over 4 years.
    # Williams %R provides mean-reversion edge in ranging markets while Donchian captures breakouts.
    # Works in bull/bear via volatility expansion/contraction cycles.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R(14) on 1d
    lookback_wr = 14
    highest_high = pd.Series(high_1d).rolling(window=lookback_wr, min_periods=lookback_wr).max().values
    lowest_low = pd.Series(low_1d).rolling(window=lookback_wr, min_periods=lookback_wr).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align 1d Williams %R to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate Donchian(20) channels on 6h
    lookback_dc = 20
    donchian_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    donchian_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(williams_r_aligned[i]) or np.isnan(vol_threshold[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_ok = volume[i] > vol_threshold[i]
        
        # Breakout conditions
        long_breakout = close[i] > donchian_high[i]
        short_breakout = close[i] < donchian_low[i]
        
        # Williams %R filters (extreme readings)
        long_filter = williams_r_aligned[i] < -80  # Oversold
        short_filter = williams_r_aligned[i] > -20  # Overbought
        
        # Entry conditions
        long_entry = long_breakout and long_filter and vol_ok and position != 1
        short_entry = short_breakout and short_filter and vol_ok and position != -1
        
        # Exit conditions: price crosses Donchian midpoint OR Williams %R returns to neutral
        exit_long = close[i] < donchian_mid[i] or williams_r_aligned[i] > -50
        exit_short = close[i] > donchian_mid[i] or williams_r_aligned[i] < -50
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
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

name = "6h_1d_donchian_williamsr_volume_v1"
timeframe = "6h"
leverage = 1.0