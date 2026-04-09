#!/usr/bin/env python3
# 12h_camarilla_1d_trend_volume_v1
# Hypothesis: 12h strategy using Camarilla pivot levels from 1d for trend structure,
# volume confirmation, and discrete position sizing (±0.30). Long when price > H3 and
# volume confirmed, short when price < L3 and volume confirmed. Exit on opposite
# H4/L4 touch or volume drop. Designed for low trade frequency (12-37/year) to
# minimize fee drag and work in both bull and bear markets via mean-reversion
# at extreme intraday levels.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_1d_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla pivot levels (based on previous 1d bar)
    # H4 = close + 1.5*(high - low)
    # H3 = close + 1.1*(high - low)
    # L3 = close - 1.1*(high - low)
    # L4 = close - 1.5*(high - low)
    hl_range = high_1d - low_1d
    H4 = close_1d + 1.5 * hl_range
    H3 = close_1d + 1.1 * hl_range
    L3 = close_1d - 1.1 * hl_range
    L4 = close_1d - 1.5 * hl_range
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(H4_aligned[i]) or np.isnan(H3_aligned[i]) or 
            np.isnan(L3_aligned[i]) or np.isnan(L4_aligned[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price touches L4 (mean reversion) OR volume drops below average
            if close[i] <= L4_aligned[i] or volume[i] < volume_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price touches H4 (mean reversion) OR volume drops below average
            if close[i] >= H4_aligned[i] or volume[i] < volume_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat
            # Need volume confirmation
            volume_confirmed = volume[i] > 2.0 * volume_ma[i]
            
            if volume_confirmed:
                # Long: price above H3
                if close[i] > H3_aligned[i]:
                    position = 1
                    signals[i] = 0.30
                # Short: price below L3
                elif close[i] < L3_aligned[i]:
                    position = -1
                    signals[i] = -0.30
    
    return signals