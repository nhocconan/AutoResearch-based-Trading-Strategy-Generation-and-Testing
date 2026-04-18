#!/usr/bin/env python3
"""
4h_12h_1d_Trix_Volume_Trend
Hypothesis: TRIX (triple exponential average momentum) on 12h timeframe identifies momentum shifts, 
combined with volume confirmation on 4h and price position relative to 1d VWAP for entry timing. 
In bull markets: long when TRIX turns positive with volume and price above VWAP. 
In bear markets: short when TRIX turns negative with volume and price below VWAP. 
Uses 1d VWAP as dynamic support/resistance to avoid whipsaws. Targets 20-30 trades/year by 
requiring TRIX zero-cross, volume > 1.5x 20-period average, and price on correct side of VWAP.
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
    
    # Get 12h data for TRIX calculation
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate TRIX: triple EMA of close, then percent change
    # EMA1
    ema1 = np.full_like(close_12h, np.nan)
    ema1[0] = close_12h[0]
    for i in range(1, len(close_12h)):
        ema1[i] = 0.15 * close_12h[i] + 0.85 * ema1[i-1]  # alpha = 2/(12+1) for 12-period EMA
    
    # EMA2 of EMA1
    ema2 = np.full_like(close_12h, np.nan)
    ema2[0] = ema1[0]
    for i in range(1, len(close_12h)):
        ema2[i] = 0.15 * ema1[i] + 0.85 * ema2[i-1]
    
    # EMA3 of EMA2
    ema3 = np.full_like(close_12h, np.nan)
    ema3[0] = ema2[0]
    for i in range(1, len(close_12h)):
        ema3[i] = 0.15 * ema2[i] + 0.85 * ema3[i-1]
    
    # TRIX = (EMA3 - previous EMA3) / previous EMA3 * 100
    trix = np.full_like(close_12h, np.nan)
    for i in range(1, len(close_12h)):
        if ema3[i-1] != 0:
            trix[i] = (ema3[i] - ema3[i-1]) / ema3[i-1] * 100
    
    # Align TRIX to 4h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_12h, trix)
    
    # Get 1d data for VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate typical price and VWAP for each 1d bar
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    vwap_1d = np.full_like(close_1d, np.nan)
    cum_tp_vol = 0.0
    cum_vol = 0.0
    
    for i in range(len(close_1d)):
        cum_tp_vol += typical_price_1d[i] * volume_1d[i]
        cum_vol += volume_1d[i]
        if cum_vol != 0:
            vwap_1d[i] = cum_tp_vol / cum_vol
        else:
            vwap_1d[i] = typical_price_1d[i]
    
    # Align VWAP to 4h timeframe
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(trix_aligned[i]) or np.isnan(vwap_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: TRIX turns positive (above zero), with volume, and price above VWAP
            if (trix_aligned[i] > 0 and trix_aligned[i-1] <= 0 and vol_confirm[i] and 
                close[i] > vwap_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: TRIX turns negative (below zero), with volume, and price below VWAP
            elif (trix_aligned[i] < 0 and trix_aligned[i-1] >= 0 and vol_confirm[i] and 
                  close[i] < vwap_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: TRIX turns negative or price falls below VWAP
            if (trix_aligned[i] < 0 or 
                close[i] < vwap_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: TRIX turns positive or price rises above VWAP
            if (trix_aligned[i] > 0 or 
                close[i] > vwap_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_12h_1d_Trix_Volume_Trend"
timeframe = "4h"
leverage = 1.0