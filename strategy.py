#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using 1d Williams %R with volume confirmation and ATR trailing stop.
Long when 1d Williams %R crosses above -80 (oversold reversal) AND volume > 1.5x 20-period average.
Short when 1d Williams %R crosses below -20 (overbought reversal) AND volume > 1.5x 20-period average.
Exit when price retraces 50% of the move from entry OR ATR trailing stop (3.0*ATR) is hit.
Uses discrete position sizing (0.25) to balance return and drawdown.
Designed for 4h timeframe to target 19-50 trades/year per symbol (75-200 total over 4 years).
Williams %R on 1d timeframe provides institutional-grade reversal signals with less noise than RSI.
Volume confirmation filters false reversals in choppy markets.
ATR trailing stop allows profits to run while limiting downside in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Williams %R (14-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    # Avoid division by zero
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align Williams %R to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Volume average (20-period) on 4h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
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
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        wr = williams_r_aligned[i]
        
        if position == 0:
            # Long: Williams %R crosses above -80 (oversold reversal) AND volume spike
            if i > start_idx and williams_r_aligned[i-1] <= -80 and wr > -80 and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
            # Short: Williams %R crosses below -20 (overbought reversal) AND volume spike
            elif i > start_idx and williams_r_aligned[i-1] >= -20 and wr < -20 and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
                entry_price = price
                lowest_since_entry = price
        else:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price)
            else:
                lowest_since_entry = min(lowest_since_entry, price)
            
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price retraces 50% of the move from entry
            if position == 1:
                retrace_level = entry_price + 0.5 * (highest_since_entry - entry_price)
                if price <= retrace_level:
                    exit_signal = True
            else:  # position == -1
                retrace_level = entry_price - 0.5 * (entry_price - lowest_since_entry)
                if price >= retrace_level:
                    exit_signal = True
            
            # ATR trailing stop: 3.0 * ATR from extreme point
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

name = "4H_WilliamsR_VolumeConfirmation_ATRTrailingStop"
timeframe = "4h"
leverage = 1.0