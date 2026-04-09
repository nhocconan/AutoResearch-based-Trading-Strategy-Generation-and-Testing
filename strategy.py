#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d ATR-based volatility filter
# Donchian breakouts capture momentum with clear entry/exit levels
# 1d ATR filter ensures trades only occur in sufficient volatility regimes
# Works in bull/bear: ATR adapts to changing market conditions
# Target: 75-200 total trades over 4 years (19-50/year) with discrete sizing 0.25-0.30

name = "4h_1d_donchian_atr_volatility_filter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 14-period ATR on 1d timeframe
    tr1 = np.abs(df_1d['high'].values - df_1d['low'].values)
    tr2 = np.abs(df_1d['high'].values - np.roll(df_1d['close'].values, 1))
    tr3 = np.abs(df_1d['low'].values - np.roll(df_1d['close'].values, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    
    atr_1d = np.full_like(tr, np.nan, dtype=np.float64)
    for i in range(len(tr)):
        if i < 13:
            atr_1d[i] = np.nan
        else:
            atr_1d[i] = np.mean(tr[i-13:i+1])
    
    # Align 1d ATR to 4h timeframe (wait for daily close)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(n):
        if i < 20:
            donchian_high[i] = np.nan
            donchian_low[i] = np.nan
        else:
            donchian_high[i] = np.max(high[i-20:i])
            donchian_low[i] = np.min(low[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(atr_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when 1d ATR > 0.5% of price
        volatility_filter = atr_1d_aligned[i] > 0.005 * close[i]
        
        if position == 1:  # Long position
            # Exit: price < Donchian low
            if close[i] < donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > Donchian high
            if close[i] > donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volatility filter and Donchian breakout
            if volatility_filter:
                # Long entry: price > Donchian high
                if close[i] > donchian_high[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price < Donchian low
                elif close[i] < donchian_low[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals