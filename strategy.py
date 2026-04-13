#!/usr/bin/env python3
"""
4h_1d_TrueRange_Breakout_Volume_Confirmation
Hypothesis: 4h True Range breakout with 1d volume confirmation works in both bull and bear markets.
Long when price breaks above prior 4h bar's high + True Range > 1.5x ATR(14) + 1d volume > 1.5x 20-day average.
Short when price breaks below prior 4h bar's low + True Range > 1.5x ATR(14) + 1d volume > 1.5x 20-day average.
Exit when price crosses the 4h midpoint of the breakout bar.
Targets 20-40 trades/year with volatility-based entries that capture expansion moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 4h True Range and ATR
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(true_range).rolling(window=14, min_periods=14).mean().values
    
    # 1d volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(atr[i]) or np.isnan(vol_ma_20_aligned[i]) or np.isnan(vol_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition: current 1d volume > 1.5x 20-day average
        vol_condition = vol_1d_aligned[i] > (vol_ma_20_aligned[i] * 1.5)
        
        # True Range condition: current TR > 1.5x ATR
        tr_condition = true_range[i] > (atr[i] * 1.5)
        
        # Breakout conditions: price breaks prior bar's high/low
        long_breakout = close[i] > high[i-1]
        short_breakout = close[i] < low[i-1]
        
        # Exit conditions: price crosses midpoint of breakout bar
        long_exit = close[i] < (high[i-1] + low[i-1]) / 2
        short_exit = close[i] > (high[i-1] + low[i-1]) / 2
        
        if position == 0:
            if long_breakout and vol_condition and tr_condition:
                position = 1
                signals[i] = position_size
            elif short_breakout and vol_condition and tr_condition:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            if long_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            if short_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_TrueRange_Breakout_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0