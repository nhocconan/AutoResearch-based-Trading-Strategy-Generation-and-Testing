#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1w volume spike and ATR regime filter
# In high volatility regimes (ATR ratio > 1.2): breakout above/below Donchian(20) with volume confirmation
# In low volatility regimes (ATR ratio <= 1.2): no new entries, hold existing positions until opposite breakout
# Uses discrete position sizing 0.25 to limit trades and reduce fee drag
# Works in bull/bear markets: breakout captures trends, volatility filter avoids whipsaws in low vol

name = "12h_1w_donchian_breakout_volume_atr_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate 1w ATR(20)
    tr1 = np.abs(high_1w[1:] - low_1w[:-1])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1w = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1w ATR ratio (current ATR / 50-period average ATR) for volatility regime
    atr_avg_1w = pd.Series(atr_1w).rolling(window=50, min_periods=50).mean().values
    atr_ratio_1w = np.where(atr_avg_1w > 0, atr_1w / atr_avg_1w, 1.0)
    
    # Calculate 1w average volume (20-period)
    avg_volume_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1w Donchian channels (20-period) based on prior week to avoid look-ahead
    highest_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align 1w indicators to 12h timeframe
    highest_1w_aligned = align_htf_to_ltf(prices, df_1w, highest_1w)
    lowest_1w_aligned = align_htf_to_ltf(prices, df_1w, lowest_1w)
    atr_ratio_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_ratio_1w)
    avg_volume_1w_aligned = align_htf_to_ltf(prices, df_1w, avg_volume_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_1w_aligned[i]) or np.isnan(lowest_1w_aligned[i]) or
            np.isnan(atr_ratio_1w_aligned[i]) or np.isnan(avg_volume_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility regime filter: high volatility = trade breakouts
        high_vol_regime = atr_ratio_1w_aligned[i] > 1.2
        
        if position == 1:  # Long position
            # Exit long if price breaks below lowest Donchian level
            if close[i] < lowest_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if price breaks above highest Donchian level
            if close[i] > highest_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if high_vol_regime:
                # Enter long on breakout above highest Donchian with volume confirmation
                if close[i] > highest_1w_aligned[i] and volume[i] > 2.0 * avg_volume_1w_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Enter short on breakout below lowest Donchian with volume confirmation
                elif close[i] < lowest_1w_aligned[i] and volume[i] > 2.0 * avg_volume_1w_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals