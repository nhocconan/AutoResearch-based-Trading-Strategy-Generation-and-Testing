#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla pivot levels from 1d + volume spike + chop regime filter.
    # Camarilla levels from prior 1d provide intraday support/resistance.
    # Volume spike confirms institutional participation.
    # Chop regime (CHOP > 61.8) triggers mean reversion at extremes.
    # Target: 75-200 total trades over 4 years = 19-50/year.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla, volume, and chop (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (based on prior day's range)
    # H4 = Close + 1.1*(High-Low)/2, L4 = Close - 1.1*(High-Low)/2
    camarilla_h4 = df_1d['close'] + 1.1 * (df_1d['high'] - df_1d['low']) / 2
    camarilla_l4 = df_1d['close'] - 1.1 * (df_1d['high'] - df_1d['low']) / 2
    
    # Calculate 1d volume MA(20) for spike detection
    volume_1d = df_1d['volume'].values
    volume_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d Choppiness Index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / log10(highest_high - lowest_low)) / log10(14)
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.rolling(window=14, min_periods=14).mean()
    highest_high = df_1d['high'].rolling(window=14, min_periods=14).max()
    lowest_low = df_1d['low'].rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr14.rolling(window=14, min_periods=14).sum() / 
                          (highest_high - lowest_low)) / np.log10(14)
    chop_values = chop.values
    
    # Align HTF indicators to 4h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4.values)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4.values)
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5 * 20-period MA (volume spike)
        volume_filter = volume[i] > 1.5 * volume_ma_aligned[i]
        
        # Chop regime filter: CHOP > 61.8 indicates ranging market (mean revert)
        chop_filter = chop_aligned[i] > 61.8
        
        # Mean reversion at Camarilla extremes
        touch_h4 = close[i] >= camarilla_h4_aligned[i]  # Touch/resistance
        touch_l4 = close[i] <= camarilla_l4_aligned[i]  # Touch/support
        
        # Entry conditions: touch extreme in chop regime with volume spike
        long_entry = touch_l4 and chop_filter and volume_filter
        short_entry = touch_h4 and chop_filter and volume_filter
        
        # Exit conditions: price returns to opposite Camarilla level or midpoint
        camarilla_mid = (camarilla_h4_aligned[i] + camarilla_l4_aligned[i]) / 2
        long_exit = close[i] >= camarilla_mid
        short_exit = close[i] <= camarilla_mid
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_camarilla_pivot_volume_chop_v1"
timeframe = "4h"
leverage = 1.0