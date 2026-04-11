# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
4h_1d_camarilla_breakout_volume_v2
Hypothesis: Camarilla pivot levels from 1-day timeframe provide strong intraday support/resistance.
Breakout above/below key Camarilla levels (H3/L3) with volume confirmation and volatility filter
trades in both bull and bear markets by capturing institutional breakouts.
Uses 1-day Camarilla levels, 4-hour price action for entry, volume confirmation, and ATR-based volatility filter.
Designed for moderate trade frequency (~25-40 trades/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for previous day
    # Using previous day's OHLC to avoid look-ahead
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    h3 = pivot + (range_hl * 1.1 / 2)
    l3 = pivot - (range_hl * 1.1 / 2)
    h4 = pivot + (range_hl * 1.1)
    l4 = pivot - (range_hl * 1.1)
    
    # Align Camarilla levels to 4h
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # Volume confirmation: 1-day volume > 20-period average
    volume_1d = df_1d['volume'].values
    vol_avg_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    # 4h ATR for volatility filter (avoid low volatility chop)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr_4h = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_4h[0] = tr1[0]  # First period TR
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    atr_avg_20 = pd.Series(atr_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from index 20 to ensure sufficient data
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(vol_avg_20_aligned[i]) or np.isnan(atr_4h[i]) or 
            np.isnan(atr_avg_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Current 1-day volume (aligned)
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        vol_confirm = vol_1d_current > vol_avg_20_aligned[i]
        
        # Volatility filter: only trade when ATR > 20-period average
        vol_filter = atr_4h[i] > atr_avg_20[i]
        
        price = close[i]
        h3_val = h3_aligned[i]
        l3_val = l3_aligned[i]
        h4_val = h4_aligned[i]
        l4_val = l4_aligned[i]
        
        # Entry conditions: Breakout of H3/L3 with volume and volatility
        long_signal = vol_confirm and vol_filter and (price > h3_val)
        short_signal = vol_confirm and vol_filter and (price < l3_val)
        
        # Exit conditions: Reversal to opposite H3/L3 or touch of H4/L4
        long_exit = price < l3_val or price > h4_val
        short_exit = price > h3_val or price < l4_val
        
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
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
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals