#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian channel breakout for direction and 1d volume spike for confirmation.
# Long when price breaks above 4h Donchian upper channel AND 1d volume > 1.5x 20-period average.
# Short when price breaks below 4h Donchian lower channel AND 1d volume > 1.5x 20-period average.
# Exit when price returns to 4h Donchian middle (mean of upper/lower) or volume drops below average.
# Uses discrete position size 0.20. 4h Donchian provides structural breakout signals, 1d volume confirms institutional participation.
# 1h timeframe for precise entry timing, targeting 60-150 total trades over 4 years (15-37/year) to minimize fee drag.
# Works in bull markets (capture breakouts) and bear markets (capture breakdowns) with volume filter to avoid false signals.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data once before loop for Donchian channel
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Get 1d data once before loop for volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # === 4h Indicators: Donchian Channel (20-period) ===
    # Upper = max(high, 20), Lower = min(low, 20), Middle = (upper + lower) / 2
    high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    upper_20 = high_20
    lower_20 = low_20
    middle_20 = (upper_20 + lower_20) / 2.0
    
    # === 1d Indicators: Volume Average (20-period) ===
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to primary timeframe (1h)
    upper_20_aligned = align_htf_to_ltf(prices, df_4h, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_4h, lower_20)
    middle_20_aligned = align_htf_to_ltf(prices, df_4h, middle_20)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 50  # Donchian 20 needs sufficient warmup + 1d vol MA
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(middle_20_aligned[i]) or np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        upper = upper_20_aligned[i]
        lower = lower_20_aligned[i]
        middle = middle_20_aligned[i]
        vol_ma = vol_ma_20_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when price returns to middle OR volume drops below average
            if (price <= middle) or (vol < vol_ma):
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when price returns to middle OR volume drops below average
            if (price >= middle) or (vol < vol_ma):
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above upper channel AND volume > 1.5x average
            if (price > upper) and (vol > 1.5 * vol_ma):
                signals[i] = 0.20
                position = 1
            
            # SHORT: Price breaks below lower channel AND volume > 1.5x average
            elif (price < lower) and (vol > 1.5 * vol_ma):
                signals[i] = -0.20
                position = -1
        
        else:
            signals[i] = position * 0.20  # maintain position
    
    return signals

name = "1h_4hDonchianBreakout_1dVolumeSpike_V1"
timeframe = "1h"
leverage = 1.0