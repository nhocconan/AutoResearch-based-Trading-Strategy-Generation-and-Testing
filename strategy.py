#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Donchian breakout with 1d volume confirmation and ATR filter
    # Enter long when price breaks above 20-bar Donchian high with volume > 1.5x 20-bar avg
    # Enter short when price breaks below 20-bar Donchian low with volume > 1.5x 20-bar avg
    # Exit when price crosses the 12h midpoint (average of Donchian levels)
    # Uses 1d HTF for volume confirmation to reduce noise and false breakouts
    # ATR filter ensures sufficient volatility for breakout validity
    # Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for primary timeframe
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Get 1d data for HTF volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # Calculate 12h Donchian channels (20-period)
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Align 1d volume to 12h timeframe
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    # Volume confirmation: volume > 1.5x 20-bar average volume (12h)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * avg_volume)
    
    # ATR filter for volatility (12h ATR > 12h ATR 50-period MA)
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    volatility_filter = atr > atr_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(1, n):  # start from 1 to access previous bar
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i]) or
            np.isnan(volume_1d_aligned[i]) or np.isnan(avg_volume[i]) or np.isnan(atr[i]) or np.isnan(atr_ma[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > donchian_high[i]  # break above upper band
        breakout_down = close[i] < donchian_low[i]  # break below lower band
        
        # Entry conditions with volume confirmation and volatility filter
        long_entry = breakout_up and volume_confirmed[i] and volatility_filter[i] and position != 1
        short_entry = breakout_down and volume_confirmed[i] and volatility_filter[i] and position != -1
        
        # Exit conditions
        exit_long = (position == 1 and close[i] < donchian_mid[i])
        exit_short = (position == -1 and close[i] > donchian_mid[i])
        
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

name = "12h_1d_donchian_volume_volatility_filter_v1"
timeframe = "12h"
leverage = 1.0