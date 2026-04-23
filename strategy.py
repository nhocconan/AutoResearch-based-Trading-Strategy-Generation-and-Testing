#!/usr/bin/env python3
"""
Hypothesis: 12h strategy using 1w Camarilla H3/L3 breakout with volume confirmation and ATR trailing stop.
Long when price breaks above 1w Camarilla H3 AND volume > 1.8x 30-period average.
Short when price breaks below 1w Camarilla L3 AND volume > 1.8x 30-period average.
Exit when price retraces 50% of the breakout move OR ATR trailing stop (3.0*ATR) hit.
Uses discrete position sizing (0.25) to limit fee drag and drawdown.
Designed for 12h timeframe targeting 12-25 trades/year per symbol (50-100 total over 4 years).
Works in bull/bear markets via volume confirmation filtering false breakouts and ATR trailing stops locking in profit.
1w Camarilla levels provide major institutional support/resistance from higher timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Camarilla levels (based on previous week's OHLC)
    camarilla_h1 = (high_1w + low_1w + close_1w) / 3.0
    camarilla_l1 = (high_1w + low_1w + close_1w) / 3.0
    camarilla_range = high_1w - low_1w
    
    camarilla_h3 = camarilla_h1 + camarilla_range * 1.1 / 4.0
    camarilla_l3 = camarilla_l1 - camarilla_range * 1.1 / 4.0
    camarilla_pivot = camarilla_h1  # Pivot point
    
    # Align Camarilla levels to 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1w, camarilla_pivot)
    
    # Volume average (30-period) on 12h timeframe
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    # ATR(14) for trailing stop calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start from index where all indicators are ready
    start_idx = max(30, 14)
    
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
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
            # Short: Price breaks below 1w Camarilla L3 AND volume spike
            elif (price < l3 and volume[i] > 1.8 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
                lowest_since_entry = price
        else:
            # Update trailing extremes
            if position == 1:
                highest_since_entry = max(highest_since_entry, price)
            elif position == -1:
                lowest_since_entry = min(lowest_since_entry, price)
            
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price retraces 50% of breakout move from entry to H3/L3
            if position == 1:
                breakout_range = h3 - entry_price
                retracement_level = entry_price + 0.5 * breakout_range
                if price <= retracement_level:
                    exit_signal = True
            elif position == -1:
                breakout_range = entry_price - l3
                retracement_level = entry_price - 0.5 * breakout_range
                if price >= retracement_level:
                    exit_signal = True
            
            # ATR-based trailing stop: 3.0 * ATR from extreme
            if position == 1 and price < highest_since_entry - 3.0 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 3.0 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Camarilla_H3L3_VolumeConfirmation_ATRTrailingStop"
timeframe = "12h"
leverage = 1.0