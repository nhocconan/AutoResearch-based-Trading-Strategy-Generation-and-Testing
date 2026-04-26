#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_Reversal_1dTrend_VolumeFilter
Hypothesis: Trade reversals at Camarilla pivot levels (H4/L4) with 1d trend filter and volume confirmation. 
Works in both bull and bear markets by fading extreme intraday moves that violate the daily trend. 
Uses tight entry conditions (H4/L4 breakouts) to keep trade frequency low (<50/year) and avoid fee drag. 
Focus on BTC/ETH as primary symbols.
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
    
    # Get 1d data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d EMA20 for trend filter (faster than EMA34 for more signals)
    close_1d = df_1d['close'].values
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values
    
    # Camarilla H4 and L4 levels (extreme reversal levels)
    H4 = close_1d_prev + (high_1d - low_1d) * 1.1 / 2
    L4 = close_1d_prev - (high_1d - low_1d) * 1.1 / 2
    
    # Align Camarilla levels
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    # Volume confirmation: 1.5x average volume (moderate to balance signals)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of 1d EMA (20), volume MA (20), ATR (14)
    start_idx = max(20, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_20_1d_aligned[i]) or 
            np.isnan(H4_aligned[i]) or 
            np.isnan(L4_aligned[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(atr[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        ema_20_1d_val = ema_20_1d_aligned[i]
        H4_val = H4_aligned[i]
        L4_val = L4_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: price breaks below L4 (oversold) with volume confirmation and uptrend bias
            long_signal = (low_val < L4_val) and (volume_val > 1.5 * vol_ma_val) and (close_val > ema_20_1d_val)
            # Short: price breaks above H4 (overbought) with volume confirmation and downtrend bias
            short_signal = (high_val > H4_val) and (volume_val > 1.5 * vol_ma_val) and (close_val < ema_20_1d_val)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price reverts to daily trend or ATR stop
            if (close_val > ema_20_1d_val or 
                close_val < entry_price - 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price reverts to daily trend or ATR stop
            if (close_val < ema_20_1d_val or 
                close_val > entry_price + 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_Pivot_Reversal_1dTrend_VolumeFilter"
timeframe = "4h"
leverage = 1.0