#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d volume regime filter and ATR-based position sizing.
Long when price breaks above 20-period Donchian high AND 1d volume > 1.2x 20-period average AND ATR(14) < 0.03 * close (low volatility breakout).
Short when price breaks below 20-period Donchian low AND 1d volume > 1.2x 20-period average AND ATR(14) < 0.03 * close.
Exit when price touches the opposite Donchian level.
Uses 1d HTF for volume regime to ensure breakouts occur during elevated participation (works in both bull and bear markets).
Target: 75-200 total trades over 4 years (19-50/year).
Donchian breakouts capture strong momentum; volume filter ensures institutional participation; low volatility filter avoids false breakouts.
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
    
    # Calculate 1d ATR for volatility filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range and ATR
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d ATR to 4h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 1d volume average for regime filter
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h ATR for volatility condition
    tr_4h1 = np.abs(high - low)
    tr_4h2 = np.abs(high - np.roll(close, 1))
    tr_4h3 = np.abs(low - np.roll(close, 1))
    tr_4h1[0] = 0
    tr_4h2[0] = 0
    tr_4h3[0] = 0
    tr_4h = np.maximum(tr_4h1, np.maximum(tr_4h2, tr_4h3))
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14 + 13 + 13)  # donchian (20), ATR calculation (14+13+13)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(atr_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        atr_1d_val = atr_1d_aligned[i]
        vol_ma_1d_val = vol_ma_1d_aligned[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        atr_4h_val = atr_4h[i]
        vol_4h_val = volume[i]
        
        # Low volatility condition: ATR(14) < 3% of price
        low_volatility = atr_4h_val < 0.03 * price
        
        if position == 0:
            # Long: Break above Donchian high AND 1d volume > 1.2x average AND low volatility
            if price > upper and df_1d['volume'].iloc[-1] > 1.2 * vol_ma_1d_val and low_volatility:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low AND 1d volume > 1.2x average AND low volatility
            elif price < lower and df_1d['volume'].iloc[-1] > 1.2 * vol_ma_1d_val and low_volatility:
                signals[i] = -0.25
                position = -1
        else:
            # Exit when price touches opposite Donchian level
            if position == 1 and price < lower:  # Long exit at Donchian low
                exit_signal = True
            elif position == -1 and price > upper:  # Short exit at Donchian high
                exit_signal = True
            else:
                exit_signal = False
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_Breakout_1dVolRegime_LowVol_Filter"
timeframe = "4h"
leverage = 1.0