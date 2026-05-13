#!/usr/bin/env python3
"""
12h_Pivot_Point_Reversal
Hypothesis: Price reversals at daily pivot points (PP, R1, S1) with volume confirmation and RSI filter work in both bull and bear markets. Go long when price bounces above S1 with volume > 1.3x average and RSI < 40, short when price rejects below R1 with volume > 1.3x average and RSI > 60. Uses 12h timeframe to limit trades (target: 15-35/year) and avoid fee drag. Pivot levels act as natural support/resistance, effective in ranging and trending markets alike.
"""

name = "12h_Pivot_Point_Reversal"
timeframe = "12h"
leverage = 1.0

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
    volume = prices['volume'].values
    
    # Get daily data for pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points: PP = (H + L + C)/3, R1 = 2*PP - L, S1 = 2*PP - H
    pp = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pp - low_1d
    s1 = 2 * pp - high_1d
    
    # Align pivot levels to 12h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Calculate RSI(14) on 12h closes
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral when undefined
    
    # Calculate volume average (20-period) for volume confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(rsi[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        vol_confirm = volume[i] > 1.3 * vol_ma_20[i]
        
        if position == 0:
            # LONG: Price crosses above S1 with volume confirmation and RSI < 40 (oversold bounce)
            if close[i-1] <= s1_aligned[i-1] and close[i] > s1_aligned[i] and vol_confirm and rsi[i] < 40:
                signals[i] = 0.25
                position = 1
            # SHORT: Price crosses below R1 with volume confirmation and RSI > 60 (overbought rejection)
            elif close[i-1] >= r1_aligned[i-1] and close[i] < r1_aligned[i] and vol_confirm and rsi[i] > 60:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below PP or RSI > 70 (overbought)
            if close[i] < pp_aligned[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above PP or RSI < 30 (oversold)
            if close[i] > pp_aligned[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals