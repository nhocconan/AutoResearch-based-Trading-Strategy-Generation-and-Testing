#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Williams %R for mean reversion with volume confirmation.
# Long when Williams %R falls below -80 (oversold) and volume > 1.5x average.
# Short when Williams %R rises above -20 (overbought) and volume > 1.5x average.
# Exit when Williams %R crosses above -50 for longs or below -50 for shorts.
# Uses Williams %R to capture mean reversion in both bull and bear markets, targeting 12-37 trades per year.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period Williams %R
    williams_r = np.full(len(close_1d), np.nan)
    for i in range(13, len(close_1d)):
        highest_high = np.max(high_1d[i - 13:i + 1])
        lowest_low = np.min(low_1d[i - 13:i + 1])
        if highest_high != lowest_low:
            williams_r[i] = (highest_high - close_1d[i]) / (highest_high - lowest_low) * -100
        else:
            williams_r[i] = -50  # neutral if no range
    
    # Align Williams %R to 12h timeframe (no extra delay needed as it's based on completed 1d bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Volume MA for confirmation (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i - 19:i + 1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Williams %R and volume MA20
    start_idx = max(13, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        wr = williams_r_aligned[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: require volume above average
        vol_filter = vol_now > 1.5 * vol_avg
        
        if position == 0:
            # Long: Williams %R below -80 (oversold) with volume filter
            if wr < -80 and vol_filter:
                signals[i] = size
                position = 1
            # Short: Williams %R above -20 (overbought) with volume filter
            elif wr > -20 and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R crosses above -50
            if wr > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Williams %R crosses below -50
            if wr < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_WilliamsR14_OversoldOverbought_Volume"
timeframe = "12h"
leverage = 1.0