#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA34 trend filter and volume confirmation.
Long when price breaks above Camarilla R1 AND close > 4h EMA34 AND volume > 1.5x 20-period average.
Short when price breaks below Camarilla S1 AND close < 4h EMA34 AND volume > 1.5x 20-period average.
Exit when price retraces to Camarilla H4/L4 level.
Uses discrete position sizing (0.20) and session filter (08-20 UTC) to target 15-37 trades/year.
1h timeframe allows precise entry timing while 4h trend filter reduces false signals in ranging markets.
Camarilla levels derived from prior day's OHLC provide institutional support/resistance.
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
    
    # Calculate 4h EMA34 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # Calculate Camarilla levels from 1d timeframe (based on prior day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    rng = high_1d - low_1d
    h1 = close_1d + 1.1 * rng * 1.0 / 12.0  # R1
    l1 = close_1d - 1.1 * rng * 1.0 / 12.0  # S1
    h4 = close_1d + 1.1 * rng * 1.1 / 2.0   # H4
    l4 = close_1d - 1.1 * rng * 1.1 / 2.0   # L4
    
    # Align Camarilla levels to 1h timeframe
    h1_aligned = align_htf_to_ltf(prices, df_1d, h1)
    l1_aligned = align_htf_to_ltf(prices, df_1d, l1)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # open_time is already datetime64[ms]
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # EMA34 needs 34, vol MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_4h_aligned[i]) or 
            np.isnan(h1_aligned[i]) or np.isnan(l1_aligned[i]) or 
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        ema34_val = ema34_4h_aligned[i]
        h1 = h1_aligned[i]
        l1 = l1_aligned[i]
        h4 = h4_aligned[i]
        l4 = l4_aligned[i]
        
        if position == 0:
            # Long: Break above Camarilla R1 AND uptrend (price > EMA34) AND volume spike (1.5x avg)
            if close[i] > h1 and close[i] > ema34_val and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.20
                position = 1
            # Short: Break below Camarilla S1 AND downtrend (price < EMA34) AND volume spike (1.5x avg)
            elif close[i] < l1 and close[i] < ema34_val and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price retraces to Camarilla H4/L4 level
            if position == 1 and close[i] <= h4:
                exit_signal = True
            elif position == -1 and close[i] >= l4:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Camarilla_R1S1_Breakout_4hEMA34_Trend_VolumeConfirmation_H4L4Exit"
timeframe = "1h"
leverage = 1.0