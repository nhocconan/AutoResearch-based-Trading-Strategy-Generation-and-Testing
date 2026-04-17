#!/usr/bin/env python3
"""
Hypothesis: On the 6-hour timeframe, price respects 1-week and 1-day key support/resistance levels.
We use 1-week high/low with a 1-day EMA34 trend filter and volume confirmation to capture breakouts.
Long when price breaks above prior 1-week high with volume > 1.5x average and price above 1-day EMA34.
Short when price breaks below prior 1-week low with volume > 1.5x average and price below 1-day EMA34.
Exit when price returns to the prior 1-week midpoint or on opposite breakout.
Designed for 6h to work in trending (breakouts) and ranging (mean reversion to mid-point) markets with ~15-25 trades per year.
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
    
    # Get 1w data for prior period's high/low
    df_1w = get_htf_data(prices, '1w')
    
    # Prior 1w high and low (use shift(1) to avoid look-ahead: use completed period's levels)
    pwhigh = df_1w['high'].shift(1).values
    pwlow = df_1w['low'].shift(1).values
    pwclose = df_1w['close'].values
    
    # Prior 1w midpoint for mean reversion exit
    pwmid = (pwhigh + pwlow) / 2
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    pclose_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 for trend filter (use prior period's close to avoid look-ahead)
    ema_34_1d = pd.Series(pclose_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all levels to 6h timeframe (waits for bars to close)
    pwhigh_6h = align_htf_to_ltf(prices, df_1w, pwhigh)
    pwlow_6h = align_htf_to_ltf(prices, df_1w, pwlow)
    pwmid_6h = align_htf_to_ltf(prices, df_1w, pwmid)
    ema_34_6h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: 20-period volume MA on 6h
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for EMA34 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(pwhigh_6h[i]) or np.isnan(pwlow_6h[i]) or np.isnan(pwmid_6h[i]) or
            np.isnan(ema_34_6h[i]) or np.isnan(volume_ma_20.iloc[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        
        if position == 0:
            # Long: price breaks above prior 1w high with volume spike and above 1d EMA34
            if price > pwhigh_6h[i] and vol > 1.5 * vol_ma and price > ema_34_6h[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below prior 1w low with volume spike and below 1d EMA34
            elif price < pwlow_6h[i] and vol > 1.5 * vol_ma and price < ema_34_6h[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to prior 1w midpoint (mean reversion) OR breaks below prior 1w low (invalidates breakout)
            if price < pwmid_6h[i] or price < pwlow_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to prior 1w midpoint (mean reversion) OR breaks above prior 1w high (invalidates breakout)
            if price > pwmid_6h[i] or price > pwhigh_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Prior1W_HL_Breakout_MeanRev"
timeframe = "6h"
leverage = 1.0