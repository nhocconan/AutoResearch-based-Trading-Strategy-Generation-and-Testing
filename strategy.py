#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_Volume_ATRFilter_V1
Hypothesis: Use daily Camarilla pivot levels (R1/S1) for breakout signals, confirmed by volume spike and ATR-based volatility filter. Go long when price breaks above daily R1 with volume > 1.5x 20-period average and ATR(14) > 0.5 * ATR(50), short when price breaks below daily S1 with same conditions. Uses Camarilla levels from prior day only (no look-ahead). Target: 20-40 trades/year by requiring confluence of breakout, volume, and volatility filter. Works in bull markets via breakouts and in bear via breakdowns. Designed to avoid overtrading (<400 total 4h trades) while maintaining edge in BTC/ETH.
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
    
    # Get daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day (based on previous day)
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    camarilla_R1 = np.full_like(close_1d, np.nan)
    camarilla_S1 = np.full_like(close_1d, np.nan)
    
    for i in range(1, len(close_1d)):
        if not (np.isnan(high_1d[i-1]) or np.isnan(low_1d[i-1]) or np.isnan(close_1d[i-1])):
            camarilla_R1[i] = close_1d[i-1] + 1.1 * (high_1d[i-1] - low_1d[i-1]) / 12
            camarilla_S1[i] = close_1d[i-1] - 1.1 * (high_1d[i-1] - low_1d[i-1]) / 12
    
    # Align Camarilla levels to 4h timeframe (use prior day's levels)
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    # ATR-based volatility filter: ATR(14) > 0.5 * ATR(50) to avoid low-volatility whipsaws
    atr_period_short = 14
    atr_period_long = 50
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = np.full_like(close, np.nan)
    atr_50 = np.full_like(close, np.nan)
    
    if len(tr) >= atr_period_long:
        # Calculate ATR(50) first
        for i in range(atr_period_long, len(tr)):
            atr_50[i] = np.mean(tr[i - atr_period_long:i])
        # Then ATR(14)
        for i in range(atr_period_short, len(tr)):
            atr_14[i] = np.mean(tr[i - atr_period_short:i])
    
    vol_filter = (atr_14 > 0.5 * atr_50) & ~np.isnan(atr_14) & ~np.isnan(atr_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(1, vol_period, atr_period_long) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(camarilla_R1_aligned[i]) or np.isnan(camarilla_S1_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R1 + volume confirmation + volatility filter
            if close[i] > camarilla_R1_aligned[i] and volume[i] > 1.5 * vol_ma[i] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S1 + volume confirmation + volatility filter
            elif close[i] < camarilla_S1_aligned[i] and volume[i] > 1.5 * vol_ma[i] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Camarilla S1 (reversal signal)
            if close[i] < camarilla_S1_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Camarilla R1 (reversal signal)
            if close[i] > camarilla_R1_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_Volume_ATRFilter_V1"
timeframe = "4h"
leverage = 1.0