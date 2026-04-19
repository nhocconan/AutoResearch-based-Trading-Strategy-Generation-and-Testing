#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + daily ATR filter + volume confirmation.
# Breakouts with volume and low volatility (ATR contraction) yield strong moves.
# Uses 1d ATR to filter for volatility regime; only trade when ATR is below median.
# Designed for low frequency (10-25 trades/year) to minimize fee drag.
# Entry: Long when close > upper band and volume spike and ATR < median.
# Short when close < lower band and volume spike and ATR < median.
# Exit: Opposite band touch or ATR expansion above median.

name = "12h_Donchian20_ATR_Volume_Filter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily ATR(14) for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate ATR(14) on daily data
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    
    tr1 = d_high - d_low
    tr2 = np.abs(d_high - np.roll(d_close, 1))
    tr3 = np.abs(d_low - np.roll(d_close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first TR is just high-low
    atr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Median ATR for regime filter
    atr_median = np.nanmedian(atr_14[~np.isnan(atr_14)])
    
    # Align ATR to 12h timeframe
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Donchian channels (20-period) on 12h data
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume spike: volume > 1.8 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, 20)  # Ensure enough data
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(atr_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when ATR is below median (low volatility regime)
        vol_filter = atr_aligned[i] < atr_median
        
        if position == 0:
            # Long: break above upper band with volume and low volatility
            if (close[i] > highest_high[i] and 
                volume_spike[i] and 
                vol_filter):
                signals[i] = 0.25
                position = 1
            # Short: break below lower band with volume and low volatility
            elif (close[i] < lowest_low[i] and 
                  volume_spike[i] and 
                  vol_filter):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price touches lower band or ATR expands above median
            if (close[i] < lowest_low[i]) or (atr_aligned[i] >= atr_median):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price touches upper band or ATR expands above median
            if (close[i] > highest_high[i]) or (atr_aligned[i] >= atr_median):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals