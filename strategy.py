#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 Breakout with 1d ATR-based volatility filter and volume spike.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d ATR for volatility regime filter (only trade when volatility is elevated).
- Entry: Long when close breaks above H3 level AND ATR(14) > 1.5 * ATR(50) AND volume > 2.0 * 4h volume MA(20);
         Short when close breaks below L3 level AND ATR(14) > 1.5 * ATR(50) AND volume > 2.0 * 4h volume MA(20).
- Exit: Long exits when close crosses below L3 level; Short exits when close crosses above H3 level.
- Signal size: 0.25 discrete to control fee drag.
- Uses Camarilla pivot levels from prior 1d for precise S/R, volatility filter to avoid low-momentum environments,
  and volume confirmation for participation. Designed to work in both bull and bear markets by trading
  only during high-volatility breakout regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot and ATR volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need sufficient data for ATR(50)
        return np.zeros(n)
    
    # Calculate ATR(14) and ATR(50) for 1d volatility regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First bar: no previous close
    tr2[0] = high_1d[0] - close_1d[0]  # Approximation for first bar
    tr3[0] = low_1d[0] - close_1d[0]   # Approximation for first bar
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Calculate Camarilla pivot levels from prior 1d OHLC
    # Camarilla: H3 = close + 1.125*(high-low), L3 = close - 1.125*(high-low)
    camarilla_H3 = close_1d + 1.125 * (high_1d - low_1d)
    camarilla_L3 = close_1d - 1.125 * (high_1d - low_1d)
    
    # Align 1d indicators to 4h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_50_aligned = align_htf_to_ltf(prices, df_1d, atr_50)
    camarilla_H3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H3)
    camarilla_L3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L3)
    
    # Get 4h data for volume MA(20)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # ATR(50) needs 50, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(atr_14_aligned[i]) or 
            np.isnan(atr_50_aligned[i]) or 
            np.isnan(camarilla_H3_aligned[i]) or 
            np.isnan(camarilla_L3_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Volatility filter: ATR(14) > 1.5 * ATR(50) indicates elevated volatility regime
        vol_filter = atr_14_aligned[i] > 1.5 * atr_50_aligned[i]
        
        # Volume confirmation: 2.0x threshold for strict entry
        vol_confirm = curr_volume > 2.0 * vol_ma[i]
        
        if position == 0:
            # Check for entry signals
            if vol_filter and vol_confirm:
                # Long: Close breaks above H3 level
                if curr_close > camarilla_H3_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Close breaks below L3 level
                elif curr_close < camarilla_L3_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: exit when close crosses below L3 level
            if curr_close < camarilla_L3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when close crosses above H3 level
            if curr_close > camarilla_H3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dATR_Vol_Filter_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0