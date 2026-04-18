#!/usr/bin/env python3
"""
4h_Pivot_R1_S1_Breakout_Volume_And_Regime
Hypothesis: Use daily Camarilla pivot levels (R1/S1) to identify breakout points.
Go long when price breaks above R1 with volume and momentum confirmation,
short when breaks below S1. Uses 1D Camarilla levels for structure and 4H volume
and RSI for confirmation. Designed to work in both bull and bear markets by
capturing strong momentum moves. Targets 20-35 trades/year with position size 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1D data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    # R1 = Close + 1.1 * (High - Low)
    # S1 = Close - 1.1 * (High - Low)
    camarilla_r1 = close_1d + 1.1 * (high_1d - low_1d)
    camarilla_s1 = close_1d - 1.1 * (high_1d - low_1d)
    
    # Align Camarilla levels to 4h timeframe (wait for daily bar close)
    r1_4h = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Calculate volume average (20-period) for confirmation
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Calculate RSI (14-period) for momentum filter
    rsi = np.full(n, np.nan)
    if n >= 14:
        delta = np.diff(close, prepend=np.nan)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.full(n, np.nan)
        avg_loss = np.full(n, np.nan)
        for i in range(14, n):
            if i == 14:
                avg_gain[i] = np.mean(gain[1:15])
                avg_loss[i] = np.mean(loss[1:15])
            else:
                avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
                avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # need volume MA and RSI
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3 * 20-period average
        vol_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        # RSI momentum filter: not in extreme overbought/oversold
        rsi_ok = (rsi[i] > 20) and (rsi[i] < 80)
        
        if position == 0:
            # Long entry: price breaks above R1 with volume and RSI confirmation
            if close[i] > r1_4h[i] and vol_confirmed and rsi_ok:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S1 with volume and RSI confirmation
            elif close[i] < s1_4h[i] and vol_confirmed and rsi_ok:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price crosses back below S1 (opposite level)
            if close[i] < s1_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back above R1 (opposite level)
            if close[i] > r1_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Pivot_R1_S1_Breakout_Volume_And_Regime"
timeframe = "4h"
leverage = 1.0