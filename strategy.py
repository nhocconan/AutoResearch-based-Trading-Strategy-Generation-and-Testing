#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using Camarilla pivot breakout with 1d EMA trend filter and volume spike.
- Calculate Camarilla levels from previous 1d OHLC: H4, L3, L4, H3
- Enter long when price breaks above H3 with volume > 1.5x 20-period volume MA and price above 1d EMA50
- Enter short when price breaks below L3 with volume > 1.5x 20-period volume MA and price below 1d EMA50
- Exit when price crosses back to the opposite pivot level (H3 for shorts, L3 for longs)
- Fixed position size 0.25 to manage drawdown
- Uses 1d trend filter to avoid counter-trend trades
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
    
    # Get 1-day data for Camarilla calculation and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla levels from previous day's OHLC
    # H4 = Close + 1.5 * (High - Low)
    # L3 = Close - 1.0 * (High - Low)
    # L4 = Close - 1.5 * (High - Low)
    # H3 = Close + 1.0 * (High - Low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    H4 = close_1d + 1.5 * (high_1d - low_1d)
    L3 = close_1d - 1.0 * (high_1d - low_1d)
    L4 = close_1d - 1.5 * (high_1d - low_1d)
    H3 = close_1d + 1.0 * (high_1d - low_1d)
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    
    # Volume confirmation: 20-period volume MA
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # warmup for volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(volume_ma_20.iloc[i]) or 
            np.isnan(H4_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(L4_aligned[i]) or np.isnan(H3_aligned[i]) or
            np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        H4_val = H4_aligned[i]
        L3_val = L3_aligned[i]
        L4_val = L4_aligned[i]
        H3_val = H3_aligned[i]
        ema_val = ema_50_aligned[i]
        
        if position == 0:
            # Look for Camarilla level breakouts with volume confirmation and trend filter
            # Long: price breaks above H3 + volume spike + price above 1d EMA50
            if price > H3_val and vol > 1.5 * vol_ma and price > ema_val:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below L3 + volume spike + price below 1d EMA50
            elif price < L3_val and vol > 1.5 * vol_ma and price < ema_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when price crosses below L3 (opposite level)
            if price < L3_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price crosses above H3 (opposite level)
            if price > H3_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_CamarillaBreakout_Volume_1dEMA50"
timeframe = "4h"
leverage = 1.0