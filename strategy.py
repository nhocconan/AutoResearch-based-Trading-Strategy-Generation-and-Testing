#!/usr/bin/env python3
"""
6h_Pivot_R1S1_Breakout_Volume_ATR_Filter
Hypothesis: 6-hour chart with daily Pivot R1/S1 breakout, volume confirmation, and ATR filter.
- Daily Pivot points provide significant support/resistance levels for institutional traders
- Breakouts above R1 or below S1 with volume confirm institutional participation
- ATR filter ensures trades occur in sufficient volatility environments
- Works in both bull/bear markets by focusing on breakout direction rather than trend bias
- Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
"""

name = "6h_Pivot_R1S1_Breakout_Volume_ATR_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # ATR(14) for volatility filter - calculated on 6h data
    def calculate_atr(high, low, close, period=14):
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First period
        
        atr = np.full_like(tr, np.nan, dtype=np.float64)
        if len(tr) >= period:
            atr[period-1] = np.mean(tr[:period])
            for i in range(period, len(tr)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr_6h = calculate_atr(high, low, close, 14)
    
    # Daily data for Pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Pivot calculation
    ph = df_1d['high'].shift(1).values  # Previous day high
    pl = df_1d['low'].shift(1).values   # Previous day low
    pc = df_1d['close'].shift(1).values # Previous day close
    
    # Standard Pivot Point calculation
    pp = (ph + pl + pc) / 3.0           # Pivot Point
    r1 = 2 * pp - pl                    # Resistance 1
    s1 = 2 * pp - ph                    # Support 1
    r2 = pp + (ph - pl)                 # Resistance 2
    s2 = pp - (ph - pl)                 # Support 2
    
    # Align Pivot levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    # ATR filter: ATR > 0.5 * 50-period average ATR (ensures sufficient volatility)
    atr_ma = pd.Series(atr_6h).rolling(window=50, min_periods=50).mean().values
    atr_filter = atr_6h > (atr_ma * 0.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20, 50)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(volume_ma[i]) or
            np.isnan(atr_ma[i]) or np.isnan(atr_6h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with volume and sufficient volatility
            if (close[i] > r1_aligned[i] and 
                volume_confirm[i] and 
                atr_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume and sufficient volatility
            elif (close[i] < s1_aligned[i] and 
                  volume_confirm[i] and 
                  atr_filter[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below S1 or volatility drops
            if (close[i] < s1_aligned[i]) or (not atr_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above R1 or volatility drops
            if (close[i] > r1_aligned[i]) or (not atr_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals