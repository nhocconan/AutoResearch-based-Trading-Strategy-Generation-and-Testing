#!/usr/bin/env python3
# 12h_1d_camarilla_volume_v1
# Strategy: 12h Camarilla pivot with daily volume confirmation
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels provide strong intraday support/resistance.
# Long when price touches L3 with bullish volume confirmation, short when touches H3 with bearish volume.
# Uses 1d volume spike for confirmation to reduce false signals. Designed for low frequency
# (15-25 trades/year) to minimize fee drag in volatile markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values (shift by 1)
    phigh = np.roll(high_1d, 1)
    plow = np.roll(low_1d, 1)
    pclose = np.roll(close_1d, 1)
    phigh[0] = phigh[1] if len(phigh) > 1 else phigh[0]
    plow[0] = plow[1] if len(plow) > 1 else plow[0]
    pclose[0] = pclose[1] if len(pclose) > 1 else pclose[0]
    
    # Camarilla levels
    range_ = phigh - plow
    camarilla_L3 = pclose + range_ * 1.1 / 6
    camarilla_H3 = pclose + range_ * 1.1 / 2
    
    # Align to 12h timeframe
    camarilla_L3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L3)
    camarilla_H3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H3)
    
    # Volume confirmation: 1d volume spike (>1.5x 20-day average)
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean()
    vol_spike_1d = vol_1d > (vol_ma_20 * 1.5)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_L3_aligned[i]) or np.isnan(camarilla_H3_aligned[i]) or 
            np.isnan(vol_spike_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Long: price touches or goes below L3 with volume spike
        if (low[i] <= camarilla_L3_aligned[i] and vol_spike_aligned[i] > 0.5 and position != 1):
            position = 1
            signals[i] = 0.25
        # Short: price touches or goes above H3 with volume spike
        elif (high[i] >= camarilla_H3_aligned[i] and vol_spike_aligned[i] > 0.5 and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: price moves back toward mean (previous close)
        elif position == 1 and close[i] >= pclose[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] <= pclose[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals