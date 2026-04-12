#!/usr/bin/env python3
"""
6h_1w_Volume_Weighted_Average_Price_VWAP_Breakout
Hypothesis: 6h timeframe with weekly VWAP bands and volume confirmation. Weekly VWAP acts as dynamic support/resistance.
Breakouts above/below VWAP ± 2*weekly ATR with volume > 1.5x average trigger entries. Works in bull/bear markets by
using volatility-adjusted bands and volume confirmation to filter false breakouts. Targets 15-35 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_Volume_Weighted_Average_Price_VWAP_Breakout"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for VWAP and ATR
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly VWAP (volume-weighted average price)
    # VWAP = sum(price * volume) / sum(volume) over the week
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    vwap_numerator = (typical_price * df_1w['volume']).cumsum()
    vwap_denominator = df_1w['volume'].cumsum()
    vwap = (vwap_numerator / vwap_denominator).values
    
    # Calculate weekly ATR (14 period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = high_1w[0] - low_1w[0]  # first period
    tr2[0] = high_1w[0] - close_1w[0]
    tr3[0] = low_1w[0] - close_1w[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate VWAP bands: VWAP ± 2 * ATR
    vwap_upper = vwap + 2.0 * atr_1w
    vwap_lower = vwap - 2.0 * atr_1w
    
    # Align to 6h timeframe
    vwap_6h = align_htf_to_ltf(prices, df_1w, vwap)
    vwap_upper_6h = align_htf_to_ltf(prices, df_1w, vwap_upper)
    vwap_lower_6h = align_htf_to_ltf(prices, df_1w, vwap_lower)
    
    # Volume average (20 period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(vwap_6h[i]) or np.isnan(vwap_upper_6h[i]) or 
            np.isnan(vwap_lower_6h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume spike: current volume > 1.5x average
        volume_spike = volume[i] > vol_ma[i] * 1.5
        
        # Entry conditions: breakout of VWAP bands with volume
        long_entry = (close[i] > vwap_upper_6h[i]) and volume_spike
        short_entry = (close[i] < vwap_lower_6h[i]) and volume_spike
        
        # Exit conditions: return to VWAP level
        long_exit = close[i] < vwap_6h[i]
        short_exit = close[i] > vwap_6h[i]
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals