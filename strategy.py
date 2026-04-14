#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Camarilla pivot breakout with 1-day volume confirmation and chop filter.
# Camarilla levels from daily provide institutional support/resistance zones.
# Breakout above H3 or below L3 with volume > 2x average captures institutional moves.
# Choppiness index < 38.2 ensures we only trade in trending markets, avoiding whipsaws in ranges.
# Position size: 0.25 (25%). Target: 20-40 trades/year per symbol (80-160 total over 4 years).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Camarilla and chop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC
    ph = df_1d['high'].shift(1).values  # previous high
    pl = df_1d['low'].shift(1).values   # previous low
    pc = df_1d['close'].shift(1).values # previous close
    
    # Camarilla levels
    range_ = ph - pl
    h3 = pc + (range_ * 1.1 / 6)
    l3 = pc - (range_ * 1.1 / 6)
    h4 = pc + (range_ * 1.1 / 2)
    l4 = pc - (range_ * 1.1 / 2)
    
    # Align Camarilla levels to 12h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # Choppiness Index (14-period) on daily
    atr = np.maximum(np.maximum(df_1d['high'] - df_1d['low'], 
                               abs(df_1d['high'] - df_1d['close'].shift(1))),
                      abs(df_1d['low'] - df_1d['close'].shift(1)))
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    true_range_sum = pd.Series(df_1d['high'] - df_1d['low']).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(atr_sum / true_range_sum) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume confirmation: 2x average volume on 12h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(30, 20)  # Need Camarilla (2 days) + vol MA (20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or
            np.isnan(chop_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Only trade in trending markets (chop < 38.2)
        trending = chop_aligned[i] < 38.2
        
        if position == 0:
            # Enter long: break above H3 with volume
            if (close[i] > h3_aligned[i] and 
                volume[i] > 2.0 * vol_ma[i] and
                trending):
                position = 1
                signals[i] = position_size
            # Enter short: break below L3 with volume
            elif (close[i] < l3_aligned[i] and 
                  volume[i] > 2.0 * vol_ma[i] and
                  trending):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: return to H3 or break below L3
            if close[i] < h3_aligned[i] or close[i] < l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: return to L3 or break above H3
            if close[i] > l3_aligned[i] or close[i] > h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Camarilla_Volume_Chop_v1"
timeframe = "12h"
leverage = 1.0