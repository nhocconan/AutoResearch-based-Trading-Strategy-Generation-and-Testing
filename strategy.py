#!/usr/bin/env python3
"""
Hypothesis: On the 12-hour timeframe, price respects the prior week's high/low as key support/resistance levels.
We combine this with a 1-day EMA34 trend filter and volume confirmation to capture breakouts and reversals.
Long when price breaks above prior week's high with volume > 1.8x average and price above daily EMA34.
Short when price breaks below prior week's low with volume > 1.8x average and price below daily EMA34.
Exit when price returns to the prior week's midpoint (mean reversion) or on opposite breakout.
Designed for 12h to work in trending (breakouts) and ranging (mean reversion to mid-point) markets.
Focus on BTC/ETH with fewer trades to avoid fee drag.
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
    
    # Get weekly data for prior week's high/low
    df_1w = get_htf_data(prices, '1w')
    
    # Prior week's high and low (use shift(1) to avoid look-ahead: use completed week's levels)
    pweek_high = df_1w['high'].shift(1).values
    pweek_low = df_1w['low'].shift(1).values
    pweek_close = df_1w['close'].values
    
    # Prior week's midpoint for mean reversion exit
    pweek_mid = (pweek_high + pweek_low) / 2
    
    # Get daily data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1-day EMA34 for trend filter (use prior day's close to avoid look-ahead)
    ema_34 = pd.Series(pweek_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all weekly levels to 12h timeframe (waits for weekly bar to close)
    pweek_high_12h = align_htf_to_ltf(prices, df_1w, pweek_high)
    pweek_low_12h = align_htf_to_ltf(prices, df_1w, pweek_low)
    pweek_mid_12h = align_htf_to_ltf(prices, df_1w, pweek_mid)
    ema_34_12h = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume confirmation: 20-period volume MA on 12h
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for EMA34 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(pweek_high_12h[i]) or np.isnan(pweek_low_12h[i]) or np.isnan(pweek_mid_12h[i]) or
            np.isnan(ema_34_12h[i]) or np.isnan(volume_ma_20.iloc[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        
        if position == 0:
            # Long: price breaks above prior week's high with volume spike and above daily EMA34
            if price > pweek_high_12h[i] and vol > 1.8 * vol_ma and price > ema_34_12h[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below prior week's low with volume spike and below daily EMA34
            elif price < pweek_low_12h[i] and vol > 1.8 * vol_ma and price < ema_34_12h[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to prior week's midpoint (mean reversion) OR breaks below prior week's low (invalidates breakout)
            if price < pweek_mid_12h[i] or price < pweek_low_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to prior week's midpoint (mean reversion) OR breaks above prior week's high (invalidates breakout)
            if price > pweek_mid_12h[i] or price > pweek_high_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_PriorWeekHL_Breakout_MeanRev"
timeframe = "12h"
leverage = 1.0