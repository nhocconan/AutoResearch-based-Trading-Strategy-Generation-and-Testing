#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d Williams %R extreme filter and volume confirmation.
# Long when price breaks above Donchian(20) high AND 1d Williams %R < -80 (oversold) AND volume > 1.5x 20-period average.
# Short when price breaks below Donchian(20) low AND 1d Williams %R > -20 (overbought) AND volume > 1.5x 20-period average.
# Williams %R acts as a contrarian filter: buying oversold breakouts in downtrends and selling overbought breakdowns in uptrends.
# Designed to work in both bull (buy oversold bounces) and bear (sell overbought bounces) markets by fading extremes at breakout points.
# Target: 75-150 trades over 4 years (19-38/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Indicators: Donchian(20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Get 1d data once before loop for Williams %R filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:  # Need enough for Williams %R calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: Williams %R(14) ===
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high_1d - close_1d) / (highest_high_1d - lowest_low_1d) * -100
    # Replace division by zero with -50 (neutral)
    williams_r = np.where((highest_high_1d - lowest_low_1d) == 0, -50, williams_r)
    
    # Align 1d Williams %R to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 20 periods needed for Donchian/volume MA, 14 for Williams %R)
    warmup = 30
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(williams_r_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        upper_channel = highest_high[i]
        lower_channel = lowest_low[i]
        williams_r_val = williams_r_aligned[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price falls below Donchian(20) low or volume spike ends
            if price < lower_channel or not vol_spike:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price rises above Donchian(20) high or volume spike ends
            if price > upper_channel or not vol_spike:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Donchian(20) high AND Williams %R < -80 (oversold) AND volume spike
            if price > upper_channel and williams_r_val < -80 and vol_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below Donchian(20) low AND Williams %R > -20 (overbought) AND volume spike
            elif price < lower_channel and williams_r_val > -20 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_Donchian20_1dWilliamsR_VolumeSpike_V1"
timeframe = "6h"
leverage = 1.0