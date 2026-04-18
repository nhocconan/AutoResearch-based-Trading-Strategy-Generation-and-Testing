#!/usr/bin/env python3
"""
4h_12h_TRIX_ZeroCross_With_Volume_And_Trend
Hypothesis: TRIX(12) zero-cross on 12h timeframe provides early trend change signals.
Combined with 12h volume confirmation (volume > 1.5x 24-bar average) and 
price > 12h EMA34 for trend alignment. Designed to catch momentum shifts in both 
bull and bear markets with tight entry conditions to limit trades and reduce fee drag.
Target: 15-30 trades/year on 4h.
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
    
    # Get 12h data for multi-timeframe analysis
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate TRIX(12) on 12h close: triple EMA of 1-period percent change
    # TRIX = EMA(EMA(EMA(ROC, 12), 12), 12) where ROC = (close/tclose-1)*100
    close_12h = df_12h['close'].values
    if len(close_12h) < 40:  # need sufficient data for triple EMA
        return np.zeros(n)
    
    # Calculate 1-period ROC in percent
    roc = np.full_like(close_12h, np.nan)
    roc[1:] = (close_12h[1:] / close_12h[:-1] - 1) * 100
    
    # Triple EMA smoothing with period 12
    ema1 = np.full_like(roc, np.nan)
    ema2 = np.full_like(roc, np.nan)
    ema3 = np.full_like(roc, np.nan)
    
    for i in range(len(roc)):
        if i < 12:
            continue
        if np.isnan(roc[i]):
            ema1[i] = np.nan
        else:
            if i == 12 or np.isnan(ema1[i-1]):
                ema1[i] = roc[i]
            else:
                ema1[i] = (roc[i] * 2 / (12 + 1)) + (ema1[i-1] * (1 - 2 / (12 + 1)))
        
        if np.isnan(ema1[i]):
            ema2[i] = np.nan
        else:
            if i == 12 or np.isnan(ema2[i-1]):
                ema2[i] = ema1[i]
            else:
                ema2[i] = (ema1[i] * 2 / (12 + 1)) + (ema2[i-1] * (1 - 2 / (12 + 1)))
        
        if np.isnan(ema2[i]):
            ema3[i] = np.nan
        else:
            if i == 12 or np.isnan(ema3[i-1]):
                ema3[i] = ema2[i]
            else:
                ema3[i] = (ema2[i] * 2 / (12 + 1)) + (ema3[i-1] * (1 - 2 / (12 + 1)))
    
    trix_12h = ema3  # TRIX is the final triple EMA
    
    # Calculate EMA34 on 12h close for trend filter
    ema34_12h = np.full_like(close_12h, np.nan)
    for i in range(len(close_12h)):
        if i < 34:
            continue
        if i == 34 or np.isnan(ema34_12h[i-1]):
            ema34_12h[i] = close_12h[i]
        else:
            ema34_12h[i] = (close_12h[i] * 2 / (34 + 1)) + (ema34_12h[i-1] * (1 - 2 / (34 + 1)))
    
    # Volume confirmation: current volume > 1.5 x 24-period average
    vol_ma = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma[i] = np.mean(volume[i-24:i])
    vol_confirm = volume > (vol_ma * 1.5)
    
    # Align all 12h indicators to 4h timeframe
    trix_12h_aligned = align_htf_to_ltf(prices, df_12h, trix_12h)
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # need EMA34 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(trix_12h_aligned[i]) or np.isnan(ema34_12h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: TRIX crosses above zero, price above EMA34, with volume
            if (trix_12h_aligned[i] > 0 and 
                trix_12h_aligned[i-1] <= 0 and  # crossed above zero
                close[i] > ema34_12h_aligned[i] and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: TRIX crosses below zero, price below EMA34, with volume
            elif (trix_12h_aligned[i] < 0 and 
                  trix_12h_aligned[i-1] >= 0 and  # crossed below zero
                  close[i] < ema34_12h_aligned[i] and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: TRIX crosses below zero or price crosses below EMA34
            if (trix_12h_aligned[i] < 0 and trix_12h_aligned[i-1] >= 0) or \
               (close[i] < ema34_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: TRIX crosses above zero or price crosses above EMA34
            if (trix_12h_aligned[i] > 0 and trix_12h_aligned[i-1] <= 0) or \
               (close[i] > ema34_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_12h_TRIX_ZeroCross_With_Volume_And_Trend"
timeframe = "4h"
leverage = 1.0