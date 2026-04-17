#!/usr/bin/env python3
"""
Hypothesis: On 4h, price respects 1-day Camarilla pivot levels (H3, L3) as support/resistance.
We combine with volume confirmation and a 12-hour EMA34 trend filter.
Long when price crosses above H3 with volume > 1.5x average and price above EMA34.
Short when price crosses below L3 with volume > 1.5x average and price below EMA34.
Exit when price returns to the 1-day midpoint (H4/L4) or on opposite signal.
Designed for 4h to work in trending and ranging markets with ~20-50 trades per year.
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
    
    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from prior 1-day data
    # Using prior day's high, low, close to avoid look-ahead
    phigh = df_1d['high'].shift(1).values
    plow = df_1d['low'].shift(1).values
    pclose = df_1d['close'].shift(1).values
    
    # Camarilla levels
    range_val = phigh - plow
    h3 = pclose + range_val * 1.1 / 4
    l3 = pclose - range_val * 1.1 / 4
    h4 = pclose + range_val * 1.1 / 2
    l4 = pclose - range_val * 1.1 / 2
    
    # Calculate 12h EMA34 for trend filter (use prior period's close to avoid look-ahead)
    ema_34 = pd.Series(pclose).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all 1d levels to 4h timeframe (waits for 1d bar to close)
    h3_4h = align_htf_to_ltf(prices, df_1d, h3)
    l3_4h = align_htf_to_ltf(prices, df_1d, l3)
    h4_4h = align_htf_to_ltf(prices, df_1d, h4)
    l4_4h = align_htf_to_ltf(prices, df_1d, l4)
    ema_34_4h = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume confirmation: 20-period volume MA on 4h
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for EMA34 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(h3_4h[i]) or np.isnan(l3_4h[i]) or np.isnan(h4_4h[i]) or 
            np.isnan(l4_4h[i]) or np.isnan(ema_34_4h[i]) or np.isnan(volume_ma_20.iloc[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        
        if position == 0:
            # Long: price crosses above H3 with volume spike and above EMA34
            if price > h3_4h[i] and vol > 1.5 * vol_ma and price > ema_34_4h[i]:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below L3 with volume spike and below EMA34
            elif price < l3_4h[i] and vol > 1.5 * vol_ma and price < ema_34_4h[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to H4 (pivot resistance) or breaks below L3 (invalidates support)
            if price < h4_4h[i] or price < l3_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to L4 (pivot support) or breaks above H3 (invalidates resistance)
            if price > l4_4h[i] or price > h3_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_12hEMA34_Volume"
timeframe = "4h"
leverage = 1.0