#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and ATR stoploss
    # Enter long when price breaks above Donchian(20) high with volume > 1.5x 20-bar avg
    # Enter short when price breaks below Donchian(20) low with volume > 1.5x 20-bar avg
    # Exit long when price touches Donchian(20) low or ATR-based stoploss hit
    # Exit short when price touches Donchian(20) high or ATR-based stoploss hit
    # Uses 1d HTF for volume confirmation (more stable than 4h) and 4h for entry timing
    # Donchian channels provide clear structure for breakouts in both bull and bear markets
    # Volume confirmation ensures breakouts have participation to avoid false signals
    # ATR stoploss manages risk during adverse moves
    # Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for primary timeframe
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Get 1d data for volume confirmation (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate ATR(14) for stoploss
    tr1 = pd.Series(high_4h).shift(1) - pd.Series(low_4h).shift(1)
    tr2 = abs(pd.Series(high_4h).shift(1) - pd.Series(close_4h))
    tr3 = abs(pd.Series(low_4h).shift(1) - pd.Series(close_4h))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d volume confirmation to 4h timeframe
    avg_volume_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirmed_1d = volume_1d > (1.5 * avg_volume_1d)
    volume_confirmed_aligned = align_htf_to_ltf(prices, df_1d, volume_confirmed_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):  # start from 20 to ensure Donchian and ATR are ready
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i]) or
            np.isnan(atr[i]) or np.isnan(volume_confirmed_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > donchian_high[i]  # break above upper channel
        breakout_down = close[i] < donchian_low[i]  # break below lower channel
        
        # Entry conditions with volume confirmation
        long_entry = breakout_up and volume_confirmed_aligned[i] and position != 1
        short_entry = breakout_down and volume_confirmed_aligned[i] and position != -1
        
        # Exit conditions
        exit_long = (position == 1 and (close[i] < donchian_low[i] or close[i] < (donchian_mid[i] - 1.5 * atr[i])))
        exit_short = (position == -1 and (close[i] > donchian_high[i] or close[i] > (donchian_mid[i] + 1.5 * atr[i])))
        
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

name = "4h_1d_donchian_volume_atr_stop_v1"
timeframe = "4h"
leverage = 1.0