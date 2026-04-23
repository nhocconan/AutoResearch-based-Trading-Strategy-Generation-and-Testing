#!/usr/bin/env python3
"""
Hypothesis: 1d strategy using 1w Camarilla H3/L3 breakout with volume confirmation and ATR stoploss.
Long when price breaks above 1w Camarilla H3 AND volume > 1.8x 50-period average.
Short when price breaks below 1w Camarilla L3 AND volume > 1.8x 50-period average.
Exit when price retouches 1w Camarilla pivot point or ATR stoploss hit (2.5*ATR).
Uses discrete position sizing (0.30) to balance return and drawdown.
Designed for 1d timeframe to target 15-25 trades/year per symbol (60-100 total over 4 years).
Works in both bull and bear markets by using volume confirmation to filter false breakouts and ATR stops to manage risk.
1w Camarilla levels provide strong institutional support/resistance from higher timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w Camarilla levels (H3/L3 are stronger levels)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Camarilla levels (based on previous week's OHLC)
    camarilla_h3 = (high_1w + low_1w + close_1w) / 3.0 + (high_1w - low_1w) * 1.1 / 4.0
    camarilla_l3 = (high_1w + low_1w + close_1w) / 3.0 - (high_1w - low_1w) * 1.1 / 4.0
    camarilla_pivot = (high_1w + low_1w + close_1w) / 3.0  # Pivot point
    
    # Align Camarilla levels to 1d timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1w, camarilla_pivot)
    
    # Volume average (50-period) on 1d timeframe
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    # ATR(20) for stoploss calculation (wider for 1d timeframe)
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        h3 = camarilla_h3_aligned[i]
        l3 = camarilla_l3_aligned[i]
        pivot = camarilla_pivot_aligned[i]
        
        if position == 0:
            # Long: Price breaks above 1w Camarilla H3 AND volume spike
            if (price > h3 and volume[i] > 1.8 * vol_ma_val):
                signals[i] = 0.30
                position = 1
                entry_price = price
            # Short: Price breaks below 1w Camarilla L3 AND volume spike
            elif (price < l3 and volume[i] > 1.8 * vol_ma_val):
                signals[i] = -0.30
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price retouches 1w Camarilla pivot point
            if position == 1 and price <= pivot:
                exit_signal = True
            elif position == -1 and price >= pivot:
                exit_signal = True
            
            # ATR-based stoploss: 2.5 * ATR from entry (wider for 1d)
            if position == 1 and price < entry_price - 2.5 * atr_val:
                exit_signal = True
            elif position == -1 and price > entry_price + 2.5 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.30 if position == 1 else -0.30
    
    return signals

name = "1D_Camarilla_H3L3_VolumeConfirmation_ATRStop"
timeframe = "1d"
leverage = 1.0