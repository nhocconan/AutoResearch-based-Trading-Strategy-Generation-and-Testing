#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and chop regime filter
    # Enter long when price breaks above Donchian(20) high with volume > 1.5x 20-bar avg and chop < 61.8
    # Enter short when price breaks below Donchian(20) low with volume > 1.5x 20-bar avg and chop < 61.8
    # Exit when price crosses Donchian(20) midpoint
    # Uses 1d HTF for volume confirmation and chop regime (more stable than 4h)
    # Donchian breakouts capture trends, volume confirms participation, chop filter avoids whipsaws in ranging markets
    # Works in bull (continuation breaks) and bear (reversal breaks at extremes)
    # Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for primary timeframe
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Get 1d data for volume and chop regime (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 4h Donchian(20) channels
    donchian_len = 20
    donchian_high = pd.Series(high_4h).rolling(window=donchian_len, min_periods=donchian_len).max().values
    donchian_low = pd.Series(low_4h).rolling(window=donchian_len, min_periods=donchian_len).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate 1d volume confirmation: volume > 1.5x 20-bar average volume
    avg_volume_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirmed_1d = volume_1d > (1.5 * avg_volume_1d)
    
    # Calculate 1d chop regime: Chop < 61.8 = trending (favor breakouts)
    # Chop = 100 * log10(sum(ATR(1), n) / (log10(n) * (max(high, n) - min(low, n))))
    atr_1d = np.abs(high_1d - low_1d)  # True Range approximation for 1d (no gaps in daily)
    sum_tr = pd.Series(atr_1d).rolling(window=20, min_periods=20).sum().values
    max_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    min_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    denominator = np.log10(20) * (max_high - min_low)
    chop_1d = 100 * (np.log10(sum_tr) / denominator)
    chop_filter = chop_1d < 61.8  # trending regime
    
    # Align 1d indicators to 4h timeframe
    volume_confirmed_aligned = align_htf_to_ltf(prices, df_1d, volume_confirmed_1d)
    chop_filter_aligned = align_htf_to_ltf(prices, df_1d, chop_filter)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(1, n):  # start from 1 to access previous bar
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i]) or
            np.isnan(volume_confirmed_aligned[i]) or np.isnan(chop_filter_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > donchian_high[i]  # break above Donchian high
        breakout_down = close[i] < donchian_low[i]  # break below Donchian low
        
        # Entry conditions with volume confirmation and chop filter
        long_entry = breakout_up and volume_confirmed_aligned[i] and chop_filter_aligned[i] and position != 1
        short_entry = breakout_down and volume_confirmed_aligned[i] and chop_filter_aligned[i] and position != -1
        
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

name = "4h_1d_donchian_vol_chop_filter_v1"
timeframe = "4h"
leverage = 1.0