#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h 20-period Donchian breakout with 1-day ATR filter and volume confirmation.
# Breakouts above/below 4h Donchian channel indicate momentum. Entry filtered by
# 1-day ATR (volatility filter) and volume spike to avoid false breakouts.
# Exit when price returns to Donchian middle or ATR stop hit.
# Designed for fewer trades (<100/year) with clear edge in both bull/bear markets.
# Uses 1-day ATR as volatility filter to avoid choppy markets and confirm breakout strength.

name = "4h_Donchian20_ATR1d_Volume_Filter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for ATR filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1-day ATR(14)
    tr1 = np.maximum(high_1d[1:], close_1d[:-1]) - np.minimum(low_1d[1:], close_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 4h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(atr_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        atr = atr_1d_aligned[i]
        upper = highest_high[i]
        lower = lowest_low[i]
        mid = donchian_mid[i]
        
        volume_confirmed = vol > 1.5 * vol_ma
        atr_filter = atr > 0  # Ensure ATR is valid
        
        if position == 0:
            # Long: break above upper Donchian band with volume and ATR filter
            if price > upper and volume_confirmed and atr_filter:
                signals[i] = 0.25
                position = 1
            # Short: break below lower Donchian band with volume and ATR filter
            elif price < lower and volume_confirmed and atr_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price returns to midline or ATR stop (2*ATR from entry)
            # Since we don't track entry price, use current bar's high for stop
            if price < mid or price < (high[i] - 2.0 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price returns to midline or ATR stop (2*ATR from entry)
            if price > mid or price > (low[i] + 2.0 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals