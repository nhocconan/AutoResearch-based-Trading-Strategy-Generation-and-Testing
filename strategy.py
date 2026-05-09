#!/usr/bin/env python3
# Hypothesis: 6h timeframe with 12h RSI mean reversion and 1-day volume spike confirmation.
# In overbought/oversold conditions (RSI > 70 or < 30 on 12h), price tends to revert.
# Enters short when 12h RSI > 70 and 1-day volume > 1.5x 20-period average volume.
# Enters long when 12h RSI < 30 and 1-day volume > 1.5x 20-period average volume.
# Exits when RSI returns to neutral range (40-60) or volume condition fails.
# Uses volume spike to confirm genuine exhaustion rather than weak pullbacks.
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25.

name = "6h_RSI_MeanReversion_VolumeSpike"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12-hour RSI (14-period)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    close_12h = df_12h['close']
    delta = close_12h.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi_12h = 100 - (100 / (1 + rs))
    rsi_12h_values = rsi_12h.values
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h_values)
    
    # Calculate 1-day volume average (20-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume']
    vol_ma_20 = volume_1d.rolling(window=20, min_periods=20).mean()
    vol_ma_20_values = vol_ma_20.values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_values)
    
    # Current 1-day volume (aligned)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d.values)
    
    # Volume spike condition: current volume > 1.5x 20-period average
    volume_spike = volume_1d_aligned > 1.5 * vol_ma_20_aligned
    
    # RSI conditions
    rsi_overbought = rsi_12h_aligned > 70
    rsi_oversold = rsi_12h_aligned < 30
    rsi_exit = (rsi_12h_aligned >= 40) & (rsi_12h_aligned <= 60)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(rsi_12h_aligned[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(rsi_exit[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: oversold RSI + volume spike
            if rsi_oversold[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: overbought RSI + volume spike
            elif rsi_overbought[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI returns to neutral OR volume spike ends
            if rsi_exit[i] or (not volume_spike[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI returns to neutral OR volume spike ends
            if rsi_exit[i] or (not volume_spike[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals