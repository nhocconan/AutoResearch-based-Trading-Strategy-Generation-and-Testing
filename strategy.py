#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour 1-day Donchian breakout with volume confirmation and volatility filter.
# Uses the previous day's Donchian channel (20-period) as a key support/resistance level.
# Long when price breaks above prior day's Donchian high with volume > 1.5x average.
# Short when price breaks below prior day's Donchian low with volume > 1.5x average.
# Exits when price returns to the prior day's Donchian midpoint.
# The 1-day timeframe avoids intraday noise while capturing multi-day trends.
# Volume confirmation ensures institutional participation.
# Position size: 0.25 (25%) to balance return and drawdown.
# Target: ~25-40 trades per year per symbol (100-160 total over 4 years).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Donchian levels
    df_1d = get_htf_data(prices, '1d')
    
    # 1d Donchian channel (20 periods) - using prior completed day
    dc_len = 20
    if len(df_1d) < dc_len:
        return np.zeros(n)
    
    dc_high_1d = pd.Series(df_1d['high']).rolling(window=dc_len, min_periods=dc_len).max().values
    dc_low_1d = pd.Series(df_1d['low']).rolling(window=dc_len, min_periods=dc_len).min().values
    dc_mid_1d = (dc_high_1d + dc_low_1d) / 2.0
    
    # Align to 4h timeframe (values available after 1d bar closes)
    dc_high_1d_aligned = align_htf_to_ltf(prices, df_1d, dc_high_1d)
    dc_low_1d_aligned = align_htf_to_ltf(prices, df_1d, dc_low_1d)
    dc_mid_1d_aligned = align_htf_to_ltf(prices, df_1d, dc_mid_1d)
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(30, dc_len, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(dc_high_1d_aligned[i]) or 
            np.isnan(dc_low_1d_aligned[i]) or
            np.isnan(dc_mid_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Enter long: break above prior day's Donchian high + volume
            if (close[i] > dc_high_1d_aligned[i] and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Enter short: break below prior day's Donchian low + volume
            elif (close[i] < dc_low_1d_aligned[i] and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: return to prior day's Donchian midpoint
            if close[i] < dc_mid_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: return to prior day's Donchian midpoint
            if close[i] > dc_mid_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Donchian_Volume_Midpoint_v1"
timeframe = "4h"
leverage = 1.0