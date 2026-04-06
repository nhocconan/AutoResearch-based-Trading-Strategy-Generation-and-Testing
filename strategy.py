#!/usr/bin/env python3
"""
1d Donchian Breakout + Volume Confirmation + Volatility Filter
Hypothesis: Price breaking Donchian(20) high/low with above-average volume indicates strong momentum.
In bull markets, breakouts above upper band continue upward; in bear markets, breakdowns below lower band continue downward.
Volume confirmation filters false breakouts. Volatility filter (ATR) avoids ranging markets.
Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14338_1d_donchian_vol_volfilt_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for volatility filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly ATR for volatility filter (high ATR = trending market)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1w_ma = pd.Series(atr_1w).rolling(window=4, min_periods=4).mean().values  # 4-week MA of ATR
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w_ma)
    
    # Daily data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: above average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > vol_ma  # Require above average volume
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 20  # Donchian period
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ma[i]) or np.isnan(atr_1w_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: opposite band touch or volatility drop
        if position == 1:  # long position
            # Exit: touch lower Donchian band OR volatility drops below threshold
            if low[i] <= donchian_low[i] or atr_1w_aligned[i] < np.percentile(atr_1w_aligned[:i+1], 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: touch upper Donchian band OR volatility drops below threshold
            if high[i] >= donchian_high[i] or atr_1w_aligned[i] < np.percentile(atr_1w_aligned[:i+1], 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: breakout with volume confirmation
            long_breakout = high[i] >= donchian_high[i] and vol_filter[i]
            short_breakout = low[i] <= donchian_low[i] and vol_filter[i]
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals
</response>