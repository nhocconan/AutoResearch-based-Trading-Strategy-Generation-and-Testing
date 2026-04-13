#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla pivot breakout with 1d volume confirmation and ATR regime filter.
    # Camarilla levels from 1d provide precise intraday support/resistance.
    # Volume spike confirms participation. ATR filter ensures volatility expansion.
    # Target: 75-200 total trades over 4 years = 19-50/year.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots, volume and ATR (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels (using standard multipliers)
    camarilla_h4 = pivot + (range_1d * 1.1 / 2)
    camarilla_h3 = pivot + (range_1d * 1.1 / 4)
    camarilla_h2 = pivot + (range_1d * 1.1 / 6)
    camarilla_l2 = pivot - (range_1d * 1.1 / 6)
    camarilla_l3 = pivot - (range_1d * 1.1 / 4)
    camarilla_l4 = pivot - (range_1d * 1.1 / 2)
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_h2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h2)
    camarilla_l2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l2)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Calculate 1d ATR(14) for volatility regime filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - close_1d)
    tr3 = np.abs(low_1d - close_1d)
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d volume MA(20) for confirmation
    volume_1d = df_1d['volume'].values
    volume_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 4h timeframe
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(atr_aligned[i]) or np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: current ATR > 20-period mean (expanding volatility)
        atr_ma_20 = pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values
        atr_ma_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_20)
        volatility_filter = atr_aligned[i] > atr_ma_aligned[i]
        
        # Volume filter: current volume > 20-period MA
        volume_filter = volume[i] > volume_ma_aligned[i]
        
        # Camarilla breakout conditions
        breakout_long = close[i] > camarilla_h4_aligned[i]  # Break above H4
        breakout_short = close[i] < camarilla_l4_aligned[i]  # Break below L4
        
        # Entry conditions: breakout with volatility AND volume filters
        long_entry = breakout_long and volatility_filter and volume_filter
        short_entry = breakout_short and volatility_filter and volume_filter
        
        # Exit conditions: price returns to opposite Camarilla level
        long_exit = close[i] < camarilla_l4_aligned[i]
        short_exit = close[i] > camarilla_h4_aligned[i]
        
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

name = "4h_1d_camarilla_breakout_atr_volume_v1"
timeframe = "4h"
leverage = 1.0