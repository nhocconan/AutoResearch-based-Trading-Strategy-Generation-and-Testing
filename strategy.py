#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout + 12h ATR filter + volume confirmation
# Enter long when price breaks above Donchian(20) high with volume > 1.5x average and volatility expansion (ATR(12h) > 1.5x ATR(30h))
# Enter short when price breaks below Donchian(20) low with volume > 1.5x average and volatility expansion
# Uses volatility regime filter to avoid choppy markets and limit trades to 75-200 total over 4 years
# Exit when price crosses Donchian middle (mean of 20-period high-low) or volatility contracts

name = "4h_donchian_12h_atr_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period) on 4h
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_high = high_roll
    donchian_low = low_roll
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # 12h ATR for volatility regime filter
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range calculation
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr2[0] = 0  # First value has no previous close
    tr3[0] = 0  # First value has no previous close
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(30) and ATR(12) on 12h timeframe
    atr_30 = pd.Series(tr).rolling(window=30, min_periods=30).mean().values
    atr_12 = pd.Series(tr).rolling(window=12, min_periods=12).mean().values
    
    # Volatility expansion: ATR(12) > 1.5 * ATR(30)
    vol_expansion = atr_12 > (1.5 * atr_30)
    vol_expansion_aligned = align_htf_to_ltf(prices, df_12h, vol_expansion)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(vol_expansion_aligned[i]) or 
            np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price crosses below Donchian middle OR volatility contracts
            if close[i] < donchian_mid[i] or not vol_expansion_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price crosses above Donchian middle OR volatility contracts
            if close[i] > donchian_mid[i] or not vol_expansion_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout with volume and volatility expansion
            if close[i] > donchian_high[i] and volume[i] > volume_threshold[i] and vol_expansion_aligned[i]:
                # Long breakout with volume and volatility expansion
                signals[i] = 0.25
                position = 1
            elif close[i] < donchian_low[i] and volume[i] > volume_threshold[i] and vol_expansion_aligned[i]:
                # Short breakdown with volume and volatility expansion
                signals[i] = -0.25
                position = -1
    
    return signals