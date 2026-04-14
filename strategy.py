#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy combining 1d Donchian breakout with 1w volume confirmation
# Uses 1d Donchian channel breakouts for directional bias (trend following)
# Confirmed by 1w volume spike (>2x average volume) to avoid false breakouts
# Works in both bull and bear markets: breakouts capture new trends, volume filter ensures validity
# Timeframe: 4h (primary), HTF: 1d (Donchian), 1w (volume)
# Position size: 0.25 (25%) to balance return and drawdown

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Donchian Channel (20 periods)
    donch_len = 20
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Upper and lower bands
    upper_dc = pd.Series(high_1d).rolling(window=donch_len, min_periods=donch_len).max().values
    lower_dc = pd.Series(low_1d).rolling(window=donch_len, min_periods=donch_len).min().values
    
    # Align Donchian levels to 4h timeframe
    upper_dc_aligned = align_htf_to_ltf(prices, df_1d, upper_dc)
    lower_dc_aligned = align_htf_to_ltf(prices, df_1d, lower_dc)
    
    # Load 1w data ONCE for volume confirmation
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w average volume (20 periods)
    vol_1w = df_1w['volume'].values
    vol_ma_20 = pd.Series(vol_1w).rolling(window=20, min_periods=20).mean().values
    
    # Align 1w volume MA to 4h timeframe
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, donch_len + 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_dc_aligned[i]) or 
            np.isnan(lower_dc_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Volume confirmation: current volume > 2x 1w average volume
        vol_spike = vol > (2 * vol_ma_20_aligned[i])
        
        if position == 0:
            # Enter long: price breaks above upper Donchian + volume spike
            if price > upper_dc_aligned[i] and vol_spike:
                position = 1
                signals[i] = position_size
            # Enter short: price breaks below lower Donchian + volume spike
            elif price < lower_dc_aligned[i] and vol_spike:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below lower Donchian (reversal signal)
            if price < lower_dc_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above upper Donchian (reversal signal)
            if price > upper_dc_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1dDonchian_1wVol_Breakout_v1"
timeframe = "4h"
leverage = 1.0