#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Camarilla pivot levels + volume spike + volume regime filter.
# Long at L3 level with volume spike, short at H3 level with volume spike.
# Volume filter: current volume > 2x 20-period average.
# Exit when price reaches opposite H3/L3 level or closes beyond extreme levels.
# Uses 1d timeframe for pivot calculation (updated only after daily bar close).
# Target: 20-40 trades/year per symbol (80-160 total over 4 years) to minimize fee drag.
# Works in bull/bear markets by fading extremes at institutional pivot levels.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous 1d bar
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC
    prev_close = df_1d['close'].iloc[-2]
    prev_high = df_1d['high'].iloc[-2]
    prev_low = df_1d['low'].iloc[-2]
    
    # Camarilla multipliers
    H3 = prev_close + (prev_high - prev_low) * 1.1 / 6
    L3 = prev_close - (prev_high - prev_low) * 1.1 / 6
    H4 = prev_close + (prev_high - prev_low) * 1.1 / 2
    L4 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Expand to full length (same value for all bars until new 1d bar)
    H3_full = np.full(len(df_1d), H3)
    L3_full = np.full(len(df_1d), L3)
    H4_full = np.full(len(df_1d), H4)
    L4_full = np.full(len(df_1d), L4)
    
    # Align to 12h timeframe (wait for 1d bar to close)
    H3_12h = align_htf_to_ltf(prices, df_1d, H3_full)
    L3_12h = align_htf_to_ltf(prices, df_1d, L3_full)
    H4_12h = align_htf_to_ltf(prices, df_1d, H4_full)
    L4_12h = align_htf_to_ltf(prices, df_1d, L4_full)
    
    # Volume confirmation: 2x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 20
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(H3_12h[i]) or 
            np.isnan(L3_12h[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long at L3 with volume spike
            if (close[i] <= L3_12h[i] and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short at H3 with volume spike
            elif (close[i] >= H3_12h[i] and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches H3 or closes below L4
            if (close[i] >= H3_12h[i] or 
                close[i] <= L4_12h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price reaches L3 or closes above H4
            if (close[i] <= L3_12h[i] or 
                close[i] >= H4_12h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Camarilla_Volume_Spike_v1"
timeframe = "12h"
leverage = 1.0