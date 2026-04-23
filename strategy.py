#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using 1d Williams %R extreme + volume confirmation + choppiness regime filter.
Long when Williams %R < -80 (oversold) AND volume > 1.5x 20-period average AND chop > 61.8 (range).
Short when Williams %R > -20 (overbought) AND volume > 1.5x 20-period average AND chop > 61.8 (range).
Exit when price retouches 1d Williams %R midpoint (-50) or ATR stoploss hit (2.0*ATR).
Uses discrete position sizing (0.25) to balance return and drawdown.
Designed for 4h timeframe to target 19-50 trades/year per symbol (75-200 total over 4 years).
Williams %R captures mean reversion in ranging markets, volume confirmation filters weak moves,
chop filter ensures we only trade in ranging conditions where mean reversion works.
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
    if len(df_1d) < 15:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R calculation
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    # Handle division by zero
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align Williams %R to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate 1d Choppiness Index (14-period)
    atr_1d = []
    tr_1d = []
    for i in range(len(df_1d)):
        if i == 0:
            tr = high_1d[i] - low_1d[i]
        else:
            tr = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
        tr_1d.append(tr)
    
    tr_1d = np.array(tr_1d)
    atr_1d_series = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    chop_denom = highest_high_14 - lowest_low_14
    chop = np.where(chop_denom != 0, 100 * np.log10(atr_1d_series / chop_denom) / np.log10(14), 50)
    
    # Align Choppiness Index to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume average (20-period) on 4h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss calculation on 4h timeframe
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        wr = williams_r_aligned[i]
        chop_val = chop_aligned[i]
        
        # Only trade in ranging markets (chop > 61.8)
        in_range = chop_val > 61.8
        
        if position == 0:
            # Long: Williams %R oversold (< -80) AND volume spike AND ranging market
            if (wr < -80 and volume[i] > 1.5 * vol_ma_val and in_range):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Williams %R overbought (> -20) AND volume spike AND ranging market
            elif (wr > -20 and volume[i] > 1.5 * vol_ma_val and in_range):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price retouches Williams %R midpoint (-50)
            if position == 1 and wr >= -50:
                exit_signal = True
            elif position == -1 and wr <= -50:
                exit_signal = True
            
            # ATR-based stoploss: 2.0 * ATR from entry
            if position == 1 and price < entry_price - 2.0 * atr_val:
                exit_signal = True
            elif position == -1 and price > entry_price + 2.0 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_WilliamsR_VolumeConfirmation_ChopFilter_ATRStop"
timeframe = "4h"
leverage = 1.0