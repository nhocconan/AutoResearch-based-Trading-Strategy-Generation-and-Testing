#!/usr/bin/env python3
"""
4h Bollinger Band Breakout with Volume Confirmation and ATR Filter
Hypothesis: In BTC/ETH, Bollinger Band breakouts combined with volume spikes and 
ATR-based volatility filtering capture genuine momentum moves while avoiding 
false breakouts in low-volatility periods. The strategy is designed to work in 
both bull and bear markets by using volatility expansion as a signal of 
increased participation, which often precedes sustained moves. 
Target: 20-40 trades/year to minimize fee drag.
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
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2) on close
    close_series = pd.Series(close)
    basis = close_series.rolling(window=20, min_periods=20).mean()
    dev = close_series.rolling(window=20, min_periods=20).std()
    upper = basis + 2 * dev
    lower = basis - 2 * dev
    
    # ATR for volatility filter (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter: 1.8x 20-period average (higher threshold to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Warmup for BB and ATR
    
    for i in range(start_idx, n):
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(atr[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        bb_upper = upper[i]
        bb_lower = lower[i]
        atr_val = atr[i]
        vol_ok = volume_filter[i]
        
        if position == 0:
            # Long: break above upper BB with volume and sufficient volatility
            if price > bb_upper and vol_ok and atr_val > 0:
                signals[i] = 0.25
                position = 1
            # Short: break below lower BB with volume and sufficient volatility
            elif price < bb_lower and vol_ok and atr_val > 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long if price returns to middle band or volatility drops
            if price < basis[i] or atr_val < 0.5 * atr[i-1]:  # Volatility contraction
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short if price returns to middle band or volatility drops
            if price > basis[i] or atr_val < 0.5 * atr[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Bollinger_Breakout_Volume_ATR_Filter"
timeframe = "4h"
leverage = 1.0