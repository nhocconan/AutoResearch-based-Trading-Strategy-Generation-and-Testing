#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Donchian(20) + 1D Volume Spike + Chop Filter
# Hypothesis: Donchian breakouts capture strong trends in crypto. Volume spikes confirm institutional participation.
# Chop filter avoids whipsaws in ranging markets. Works in bull (breakouts up) and bear (breakouts down).
# Target: 12-37 trades/year on 12h timeframe to minimize fee drag.
name = "12h_donchian20_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for chop filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Donchian(20) on 12h
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1-day chop filter (Choppiness Index)
    # Chop = 100 * log10(sum(atr(14)) / (log10(highest_high - lowest_low))) / log10(14)
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.rolling(window=14, min_periods=14).mean()
    highest_high = df_1d['high'].rolling(window=14, min_periods=14).max()
    lowest_low = df_1d['low'].rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr14.rolling(window=14, min_periods=14).sum() / (highest_high - lowest_low)) / np.log10(14)
    chop_values = chop.values
    chop_12h = align_htf_to_ltf(prices, df_1d, chop_values)
    
    # Volume filter: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(chop_12h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below Donchian low OR chop > 61.8 (ranging)
            if close[i] <= donch_low[i] or chop_12h[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian high OR chop > 61.8 (ranging)
            if close[i] >= donch_high[i] or chop_12h[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Require volume confirmation and chop < 61.8 (trending)
            if vol_filter[i] and chop_12h[i] < 61.8:
                # Long: break above Donchian high
                if close[i] > donch_high[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: break below Donchian low
                elif close[i] < donch_low[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals