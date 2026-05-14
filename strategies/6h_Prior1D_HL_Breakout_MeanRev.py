#!/usr/bin/env python3
"""
Hypothesis: On the 6-hour timeframe, price respects the 1-day and 1-week key support/resistance levels.
We combine this with a 1-day EMA34 trend filter and volume confirmation to capture breakouts.
Long when price breaks above prior 1-day high with volume > 2x average and price above 1-day EMA34.
Short when price breaks below prior 1-day low with volume > 2x average and price below 1-day EMA34.
Exit when price returns to the prior 1-day midpoint (mean reversion) or on opposite breakout.
Designed for 6h to work in trending (breakouts) and ranging (mean reversion to mid-point) markets with ~15-30 trades per year.
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
    
    # Get 1d data for prior period's high/low and EMA34
    df_1d = get_htf_data(prices, '1d')
    
    # Prior 1d high and low (use shift(1) to avoid look-ahead: use completed period's levels)
    phigh = df_1d['high'].shift(1).values
    plow = df_1d['low'].shift(1).values
    pclose = df_1d['close'].values
    
    # Prior 1d midpoint for mean reversion exit
    pmid = (phigh + plow) / 2
    
    # Calculate 1d EMA34 for trend filter (use prior period's close to avoid look-ahead)
    ema_34 = pd.Series(pclose).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all 1d levels to 6h timeframe (waits for 1d bar to close)
    phigh_6h = align_htf_to_ltf(prices, df_1d, phigh)
    plow_6h = align_htf_to_ltf(prices, df_1d, plow)
    pmid_6h = align_htf_to_ltf(prices, df_1d, pmid)
    ema_34_6h = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume confirmation: 20-period volume MA on 6h
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for EMA34 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(phigh_6h[i]) or np.isnan(plow_6h[i]) or np.isnan(pmid_6h[i]) or
            np.isnan(ema_34_6h[i]) or np.isnan(volume_ma_20.iloc[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        
        if position == 0:
            # Long: price breaks above prior 1d high with volume spike and above 1d EMA34
            if price > phigh_6h[i] and vol > 2.0 * vol_ma and price > ema_34_6h[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below prior 1d low with volume spike and below 1d EMA34
            elif price < plow_6h[i] and vol > 2.0 * vol_ma and price < ema_34_6h[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to prior 1d midpoint (mean reversion) OR breaks below prior 1d low (invalidates breakout)
            if price < pmid_6h[i] or price < plow_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to prior 1d midpoint (mean reversion) OR breaks above prior 1d high (invalidates breakout)
            if price > pmid_6h[i] or price > phigh_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Prior1D_HL_Breakout_MeanRev"
timeframe = "6h"
leverage = 1.0