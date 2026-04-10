#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume confirmation and chop regime filter
# - Long when price breaks above Donchian(20) high AND 1d volume > 1.8x 20-period avg AND chop(14) > 61.8 (range)
# - Short when price breaks below Donchian(20) low AND 1d volume > 1.8x 20-period avg AND chop(14) > 61.8 (range)
# - Exit when price returns to Donchian(20) midpoint
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Donchian breakouts capture momentum; chop filter ensures mean reversion in ranging markets
# - Volume confirmation avoids low-liquidity false breakouts
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# - Works in both bull and bear markets: breakouts in trends, mean reversion in ranges via chop filter

name = "12h_1d_donchian_breakout_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d volume confirmation: > 1.8x 20-period average
    volume_1d = df_1d['volume'].values
    volume_20_avg = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.8 * volume_20_avg)
    
    # Align 1d volume spike to 12h timeframe
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Pre-compute Donchian(20) on 12h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_high = highest_high
    donchian_low = lowest_low
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Pre-compute Choppiness Index(14) on 12h data
    atr = pd.Series(np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))).rolling(window=14, min_periods=14).mean().values
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / np.log(14) / (highest_high_14 - lowest_low_14))
    chop = np.where((highest_high_14 - lowest_low_14) == 0, 50, chop)  # Avoid division by zero
    chop_range = chop > 61.8  # Chop > 61.8 indicates ranging market
    
    # Entry conditions
    long_entry = (close > donchian_high) & vol_spike_1d_aligned & chop_range
    short_entry = (close < donchian_low) & vol_spike_1d_aligned & chop_range
    
    # Exit condition: return to midpoint
    long_exit = np.abs(close - donchian_mid) < 0.1 * (donchian_high - donchian_low)
    short_exit = np.abs(close - donchian_mid) < 0.1 * (donchian_high - donchian_low)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(long_entry[i]) or np.isnan(short_entry[i]) or 
            np.isnan(long_exit[i]) or np.isnan(short_exit[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            if long_entry[i]:
                position = 1
                signals[i] = 0.25
            elif short_entry[i]:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to midpoint
            # Exit when price returns to Donchian midpoint
            exit_signal = long_exit[i] if position == 1 else short_exit[i]
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals