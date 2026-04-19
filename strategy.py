#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour 1-day/1-week Pivot R1/S1 breakout with volume confirmation and ATR filter.
# Long when: Price breaks above R1, volume > 1.5x 20-period average, ATR(14) > 0.5 * ATR(50)
# Short when: Price breaks below S1, volume > 1.5x 20-period average, ATR(14) > 0.5 * ATR(50)
# Exit when: Price crosses back through the pivot point (PP)
# Pivot levels provide institutional support/resistance, volume confirms breakout strength, ATR filter avoids low volatility whipsaws.
# Target: 15-30 trades/year per symbol. Works in bull (buy breakouts) and bear (sell breakdowns).
name = "12h_1d_1w_Pivot_R1S1_Breakout_Volume_ATRFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day data for pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1-week data for ATR calculation
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate daily pivot point (PP) and levels
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = close_1d + (high_1d - low_1d) * 1.1 / 12.0
    s1_1d = close_1d - (high_1d - low_1d) * 1.1 / 12.0
    
    # Calculate ATR on weekly data
    tr1 = np.maximum(high_1w - low_1w, np.abs(high_1w - np.roll(close_1w, 1)))
    tr2 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high_1w[0] - low_1w[0]  # First bar
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Align data to 12H timeframe
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    atr_14_aligned = align_htf_to_ltf(prices, df_1w, atr_14)
    atr_50_aligned = align_htf_to_ltf(prices, df_1w, atr_50)
    
    # 20-period volume average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for ATR50 calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pp_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or np.isnan(atr_14_aligned[i]) or 
            np.isnan(atr_50_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        pp = pp_1d_aligned[i]
        r1 = r1_1d_aligned[i]
        s1 = s1_1d_aligned[i]
        atr14 = atr_14_aligned[i]
        atr50 = atr_50_aligned[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # ATR filter: avoid low volatility whipsaws
        atr_filter = atr14 > 0.5 * atr50
        
        if position == 0:
            # Long entry: Price breaks above R1, volume spike, ATR filter
            if (price > r1 and close[i-1] <= r1 and 
                vol > 1.5 * vol_ma and atr_filter):
                signals[i] = 0.25
                position = 1
            # Short entry: Price breaks below S1, volume spike, ATR filter
            elif (price < s1 and close[i-1] >= s1 and 
                  vol > 1.5 * vol_ma and atr_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses back below pivot point
            if price < pp:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses back above pivot point
            if price > pp:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals