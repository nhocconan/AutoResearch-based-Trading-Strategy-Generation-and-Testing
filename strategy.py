#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using Camarilla pivot levels from 1-day with volume confirmation and choppiness regime filter.
- Calculate Camarilla levels (H4, L4, H3, L3) from previous day's OHLC
- Enter long when price crosses above H3 with volume > 2x 20-period volume MA and CHOP > 61.8 (ranging market)
- Enter short when price crosses below L3 with volume > 2x 20-period volume MA and CHOP > 61.8
- Exit when price touches opposite L3/H3 level or CHOP < 38.2 (trending market)
- Fixed position size 0.25 to manage drawdown
- Designed for 4h timeframe with strict entry conditions to limit trades to 75-200 total over 4 years
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
    
    # Get 1-day data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate previous day's OHLC for Camarilla levels
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    # Camarilla levels (based on previous day)
    H4 = prev_close + 1.5 * prev_range
    L4 = prev_close - 1.5 * prev_range
    H3 = prev_close + 1.25 * prev_range
    L3 = prev_close - 1.25 * prev_range
    
    # Align Camarilla levels to 4h timeframe
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    # Choppiness Index (using daily data for regime filter)
    # CHOP = 100 * log10(sum(ATR(14)) / (log10(n) * (highest_high - lowest_low)))
    atr_period = 14
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    atr = tr.rolling(window=atr_period, min_periods=atr_period).mean()
    
    highest_high = df_1d['high'].rolling(window=atr_period, min_periods=atr_period).max()
    lowest_low = df_1d['low'].rolling(window=atr_period, min_periods=atr_period).min()
    chop = 100 * (np.log10(atr.rolling(window=atr_period, min_periods=atr_period).sum()) / 
                  (np.log10(atr_period) * (highest_high - lowest_low)))
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop.values, additional_delay_bars=0)
    
    # Volume confirmation: 20-period volume MA
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(20, 30)  # warmup for volume MA and chop calculation
    
    for i in range(start_idx, n):
        if (np.isnan(volume_ma_20.iloc[i]) or 
            np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        chop_val = chop_aligned[i]
        
        if position == 0:
            # Look for entries in ranging market (CHOP > 61.8)
            # Long: price crosses above H3 with volume spike
            if price > H3_aligned[i] and close[i-1] <= H3_aligned[i] and vol > 2.0 * vol_ma and chop_val > 61.8:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below L3 with volume spike
            elif price < L3_aligned[i] and close[i-1] >= L3_aligned[i] and vol > 2.0 * vol_ma and chop_val > 61.8:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price touches L3 or market starts trending (CHOP < 38.2)
            if price <= L3_aligned[i] or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price touches H3 or market starts trending (CHOP < 38.2)
            if price >= H3_aligned[i] or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Volume_ChopFilter"
timeframe = "4h"
leverage = 1.0