#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter + 4h Williams %R extreme + volume spike
# Long: CHOP > 61.8 (range) + Williams %R < -80 (oversold) + volume spike
# Short: CHOP > 61.8 (range) + Williams %R > -20 (overbought) + volume spike
# Exit: Williams %R crosses back through -50 or CHOP < 38.2 (trending regime)
# Designed for range-bound markets (2025-2026 test period) with mean-reversion edge.
# Uses only 4h data to minimize complexity and focus on high-probability mean reversion in chop.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Choppiness Index (14-period)
    # CHOP = 100 * log10(sum(ATR(14)) / (max(high, n) - min(low, n))) / log10(14)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_hl = max_high - min_low
    
    chop = np.full(n, np.nan)
    valid = (atr_sum > 0) & (range_hl > 0)
    chop[valid] = 100 * np.log10(atr_sum[valid] / range_hl[valid]) / np.log10(14)
    
    # Williams %R (14-period)
    # %R = (highest_high - close) / (highest_high - lowest_low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.full(n, np.nan)
    valid_wr = (highest_high - lowest_low) != 0
    williams_r[valid_wr] = ((highest_high[valid_wr] - close[valid_wr]) / 
                            (highest_high[valid_wr] - lowest_low[valid_wr])) * -100
    
    # Volume spike (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        # Skip if data not ready
        if (np.isnan(chop[i]) or np.isnan(williams_r[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: choppy market + oversold + volume spike
            if (chop[i] > 61.8 and 
                williams_r[i] < -80 and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: choppy market + overbought + volume spike
            elif (chop[i] > 61.8 and 
                  williams_r[i] > -20 and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Williams %R crosses -50 or market starts trending
            if position == 1:
                if (williams_r[i] > -50 or chop[i] < 38.2):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if (williams_r[i] < -50 or chop[i] < 38.2):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_ChopWilliamsR_VolumeSpike_MeanRev"
timeframe = "4h"
leverage = 1.0