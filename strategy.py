#!/usr/bin/env python3
"""
Hypothesis: On 4h timeframe, price respects 1-day high/low levels as key support/resistance.
We combine 1-day high/low breakout with volume confirmation (1.5x 20-period volume MA) 
and a 1-day EMA50 trend filter. Long when price breaks above prior 1-day high with 
volume confirmation and price above EMA50. Short when price breaks below prior 1-day low 
with volume confirmation and price below EMA50. Exit on opposite breakout or when price 
returns to the prior 1-day midpoint. Designed for 4h to work in both bull (breakouts) 
and bear (mean reversion to midpoint) markets with ~20-40 trades per year.
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
    
    # Get 1d data for prior period's high/low and EMA50
    df_1d = get_htf_data(prices, '1d')
    
    # Prior 1d high and low (use shift(1) to avoid look-ahead: use completed period's levels)
    phigh = df_1d['high'].shift(1).values
    plow = df_1d['low'].shift(1).values
    pclose = df_1d['close'].values
    
    # Prior 1d midpoint for mean reversion exit
    pmid = (phigh + plow) / 2
    
    # Calculate 1d EMA50 for trend filter (use prior period's close to avoid look-ahead)
    ema_50 = pd.Series(pclose).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all 1d levels to 4h timeframe (waits for 1d bar to close)
    phigh_4h = align_htf_to_ltf(prices, df_1d, phigh)
    plow_4h = align_htf_to_ltf(prices, df_1d, plow)
    pmid_4h = align_htf_to_ltf(prices, df_1d, pmid)
    ema_50_4h = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: 20-period volume MA on 4h
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for EMA50 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(phigh_4h[i]) or np.isnan(plow_4h[i]) or np.isnan(pmid_4h[i]) or
            np.isnan(ema_50_4h[i]) or np.isnan(volume_ma_20.iloc[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        
        if position == 0:
            # Long: price breaks above prior 1d high with volume confirmation and above 1d EMA50
            if price > phigh_4h[i] and vol > 1.5 * vol_ma and price > ema_50_4h[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below prior 1d low with volume confirmation and below 1d EMA50
            elif price < plow_4h[i] and vol > 1.5 * vol_ma and price < ema_50_4h[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to prior 1d midpoint (mean reversion) OR breaks below prior 1d low (invalidates breakout)
            if price < pmid_4h[i] or price < plow_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to prior 1d midpoint (mean reversion) OR breaks above prior 1d high (invalidates breakout)
            if price > pmid_4h[i] or price > phigh_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Prior1D_HL_Breakout_MeanRev"
timeframe = "4h"
leverage = 1.0