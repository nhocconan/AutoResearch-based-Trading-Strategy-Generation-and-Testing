#!/usr/bin/env python3
"""
12h Prior Day High/Low Breakout with Volume Spike and Volume Oscillator Filter
Long: Price > prior 1D high AND volume > 1.5x 12h volume MA AND volume oscillator > 0
Short: Price < prior 1D low AND volume > 1.5x 12h volume MA AND volume oscillator < 0
Exit: Opposite break of prior 1D level
Volume oscillator: (volume - volume MA(24)) / volume MA(24) on 12h timeframe
Filters out low-momentum breakouts and adds momentum confirmation
Target: 20-30 trades/year per symbol
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1D data for prior high/low
    df_1d = get_htf_data(prices, '1d')
    prior_1d_high = df_1d['high'].shift(1)  # Prior day's high
    prior_1d_low = df_1d['low'].shift(1)    # Prior day's low
    
    prior_1d_high_aligned = align_htf_to_ltf(prices, df_1d, prior_1d_high.values)
    prior_1d_low_aligned = align_htf_to_ltf(prices, df_1d, prior_1d_low.values)
    
    # Get 12h data for volume MA and volume oscillator
    df_12h = get_htf_data(prices, '12h')
    volume_ma_24 = pd.Series(df_12h['volume']).rolling(window=24, min_periods=24).mean()
    volume_osc = (pd.Series(df_12h['volume']) - volume_ma_24) / volume_ma_24
    
    volume_ma_24_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_24.values)
    volume_osc_aligned = align_htf_to_ltf(prices, df_12h, volume_osc.values)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    entry_price = 0.0
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(prior_1d_high_aligned[i]) or np.isnan(prior_1d_low_aligned[i]) or 
            np.isnan(volume_ma_24_aligned[i]) or np.isnan(volume_osc_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_24_aligned[i]
        vol_osc = volume_osc_aligned[i]
        
        if position == 0:
            # Long: break above prior 1D high + volume spike + positive volume oscillator
            if price > prior_1d_high_aligned[i] and vol > 1.5 * vol_ma and vol_osc > 0:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: break below prior 1D low + volume spike + negative volume oscillator
            elif price < prior_1d_low_aligned[i] and vol > 1.5 * vol_ma and vol_osc < 0:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: break below prior 1D low
            if price < prior_1d_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: break above prior 1D high
            if price > prior_1d_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Prior1D_HL_Breakout_VolumeSpike_VolOsc"
timeframe = "12h"
leverage = 1.0